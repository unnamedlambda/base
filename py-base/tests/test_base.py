import struct
import pytest
from py_base import Artifact, Driver, load_artifact

import cbor

# --- constructing CLIF programs -------------------------------------------
#
# An artifact's functions are the program as data, so these build the shape
# `base_types::clif` reads: an instruction is a Cranelift format, its fields in
# the order Cranelift gives them, starting with every value it binds, the
# opcode, and the type where CLIF does not take it from an operand. Same
# vocabulary as `base/tests/common/mod.rs` on the Rust side.

def program(functions):
    return {"functions": functions}

def param(ty="i64"):
    return [ty, []]

# What a host calls a function as: the input and its length, the output
# and its length, answering a status.
BASE_FUNCTION = {"convention": "Host", "params": [param()] * 4, "returns": [param()]}

def function(blocks, name=None, answers=False):
    """A named function's signature is BASE_FUNCTION; any other function takes what
    its entry block does, and answers an i64 when `answers`."""
    signature = BASE_FUNCTION if name else {
        "convention": "Host",
        "params": [param(t) for _, t in blocks[0]["params"]],
        "returns": [param()] if answers else [],
    }
    return {"name": name, "signature": signature, "stack_slots": [], "blocks": blocks}

def block(n, params, insts):
    return {"reference": n, "params": [[p, "i64"] for p in params], "insts": insts}

def iconst64(d, k):   return {"UnaryImm": [[d], "iconst", "i64", k]}
def iconst32(d, k):   return {"UnaryImm": [[d], "iconst", "i32", k]}
def iadd(d, a, b):    return {"Binary": [[d], "iadd", None, a, b]}
def isub(d, a, b):    return {"Binary": [[d], "isub", None, a, b]}
def imul(d, a, b):    return {"Binary": [[d], "imul", None, a, b]}
def ishl(d, a, b):    return {"Binary": [[d], "ishl", None, a, b]}
def ushr(d, a, b):    return {"Binary": [[d], "ushr", None, a, b]}
def icmp(d, cc, a, b): return {"IntCompare": [[d], "icmp", None, cc, a, b]}
def store(v, addr, off=0): return {"Store": [[], "store", None, [], v, addr, off]}
def jump(t, args):    return {"Jump": [[], "jump", None, [t, args]]}
def brif(c, t, ta, e, ea): return {"Brif": [[], "brif", None, c, [t, ta], [e, ea]]}
def ret(v=None):      return {"MultiAry": [[], "return", None, [] if v is None else [v]]}

def _load(d, op, ty, addr, off, aligned=False):
    return {"Load": [[d], op, ty, ["aligned"] if aligned else [], addr, off]}

def load64(d, addr, off=0): return _load(d, "load", "i64", addr, off)
def load32(d, addr, off=0): return _load(d, "load", "i32", addr, off)

def artifact(functions, arena_size=256, data=()):
    return cbor.encode({
        "primitives": [],
        "functions": functions,
        "arena_size": arena_size,
        "data": list(data),
    })


# The name of the program's one function.
DOUBLE = "double"

# The entry block is called with the caller's input buffer and its length and
# the caller's output buffer and its length, which is the C function a host
# calls. Naming them 3, 6, 9 and 99 lets the body below refer to them the way
# it always has. Copies each i32 from data to out, multiplied by 2, and
# answers 0.

DOUBLE_I32_PROG = program([
    function([
        block(0, [0], [
            ret(),
        ]),
    ]),
    function(name=DOUBLE, blocks=[
        block(0, [3, 6, 9, 99], [
            iconst64(10, 2),
            ushr(11, 6, 10),
            iconst64(20, 0),
            jump(1, [20]),
        ]),
        block(1, [12], [
            icmp(21, "ult", 12, 11),
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
            iconst64(31, 0),
            ret(31),
        ]),
    ]),
])

def make_double_artifact():
    return artifact(DOUBLE_I32_PROG["functions"])


def pack_i32s(values):
    return struct.pack(f"<{len(values)}i", *values)


def unpack_i32s(data, count):
    return list(struct.unpack(f"<{count}i", data[:count * 4]))


class TestArtifact:
    def test_valid(self):
        assert Artifact(make_double_artifact()) is not None

    def test_json_is_not_an_artifact(self):
        """The runtime decodes the artifact, so that is where bad input is caught."""
        with pytest.raises(ValueError, match="not an artifact"):
            Driver(Artifact(b'{"primitives": [], "functions": [], "arena_size": 8, "data": []}'))

    def test_missing_fields(self):
        with pytest.raises(ValueError, match="primitives"):
            Driver(Artifact(cbor.encode({"functions": []})))

    def test_from_file(self, tmp_path):
        path = tmp_path / "double.cbor"
        path.write_bytes(make_double_artifact())
        assert Driver(load_artifact(str(path))).function(DOUBLE)() == 0


class TestBase:
    def test_new(self):
        base = Driver(Artifact(make_double_artifact()))
        assert base is not None

    def test_malformed_program(self):
        """v9 is never defined, so the program cannot be built."""
        malformed = artifact(program([function([block(0, [0], [store(9, 0), ret()])])])["functions"])
        with pytest.raises(ValueError, match="v9 used before it is defined"):
            Driver(Artifact(malformed))

    def test_call_no_data(self):
        base = Driver(Artifact(make_double_artifact()))
        # With no input there is nothing to double, and it answers 0.
        assert base.function(DOUBLE)() == 0

    def test_call_answers_a_status(self):
        """A function answers the status its `return` carries."""
        answering = artifact(program([function([block(0, [1, 2, 3, 4], [
            iconst64(5, 42), ret(5)])], name="answer")])["functions"])
        base = Driver(Artifact(answering))
        assert base.function("answer")() == 42

    def test_call_into_doubles(self):
        base = Driver(Artifact(make_double_artifact()))

        values = [1, 2, 3, 4, 5, 10, 100, -7]
        data = pack_i32s(values)
        out = bytearray(len(data))
        base.function(DOUBLE)(data, out)

        result = unpack_i32s(out, len(values))
        assert result == [v * 2 for v in values]

    def test_call_into_reuse(self):
        base = Driver(Artifact(make_double_artifact()))

        for seed in range(5):
            values = list(range(seed * 10, seed * 10 + 20))
            data = pack_i32s(values)
            out = bytearray(len(data))
            base.function(DOUBLE)(data, out)
            result = unpack_i32s(out, len(values))
            assert result == [v * 2 for v in values]

    def test_call_into_large(self):
        base = Driver(Artifact(make_double_artifact()))

        n = 100_000
        values = list(range(n))
        data = pack_i32s(values)
        out = bytearray(len(data))
        base.function(DOUBLE)(data, out)

        result = unpack_i32s(out, n)
        for i in range(n):
            assert result[i] == values[i] * 2, f"Mismatch at index {i}"

    def test_call_into_empty(self):
        base = Driver(Artifact(make_double_artifact()))
        out = bytearray(0)
        base.function(DOUBLE)(b"", out)

    def test_bytes_input(self):
        base = Driver(Artifact(make_double_artifact()))

        data = bytes(pack_i32s([42, -1, 0]))
        out = bytearray(len(data))
        base.function(DOUBLE)(data, out)
        assert unpack_i32s(out, 3) == [84, -2, 0]

    def test_call_into_answers_through_out(self):
        """The result is what the program wrote to `out`; the status is 0."""
        base = Driver(Artifact(make_double_artifact()))

        out = bytearray(16)
        assert base.function(DOUBLE)(pack_i32s([1, 2, 3, 4]), out) == 0
        assert unpack_i32s(out, 4) == [2, 4, 6, 8]

    def test_a_name_no_function_has_is_refused(self):
        base = Driver(Artifact(make_double_artifact()))
        with pytest.raises(ValueError, match='"triple"'):
            base.function("triple")()

    def test_read_answers_bytes(self):
        """What the artifact starts from, as `bytes` a host can unpack."""
        started = artifact(DOUBLE_I32_PROG["functions"], data=[{"offset": 8, "bytes": bytes([7, 0, 0, 0])}])
        base = Driver(Artifact(started))
        got = base.read(8, 4)
        assert isinstance(got, bytes)
        assert struct.unpack("<I", got)[0] == 7
        with pytest.raises(ValueError, match="outside"):
            base.read(254, 4)

    def test_the_arena_is_a_view_of_it(self):
        """The arena itself, not a copy: what a call leaves there is in the
        view, a write through the view is what the next call sees, and the
        view keeps its driver alive."""
        started = artifact(DOUBLE_I32_PROG["functions"], data=[{"offset": 8, "bytes": bytes([7, 0, 0, 0])}])
        base = Driver(Artifact(started))
        arena = base.arena
        assert len(arena) == 256 and not arena.readonly
        assert bytes(arena[8:12]) == bytes([7, 0, 0, 0])
        arena[8:12] = bytes([9, 0, 0, 0])
        assert base.read(8, 4) == bytes([9, 0, 0, 0])
        import gc
        del base
        gc.collect()
        assert bytes(arena[8:12]) == bytes([9, 0, 0, 0])


class TestFunction:
    def test_a_function_is_found_once_and_called_often(self):
        double = Driver(Artifact(make_double_artifact())).function(DOUBLE)
        for k in range(1, 4):
            out = bytearray(16)
            assert double(pack_i32s([k] * 4), out) == 0
            assert unpack_i32s(out, 4) == [2 * k] * 4

    def test_a_function_keeps_its_driver_alive(self):
        """The driver is freed only when nothing can call into it."""
        import gc
        base = Driver(Artifact(make_double_artifact()))
        double = base.function(DOUBLE)
        del base
        gc.collect()
        out = bytearray(16)
        assert double(pack_i32s([5, 6, 7, 8]), out) == 0
        assert unpack_i32s(out, 4) == [10, 12, 14, 16]
