import json
import struct
import pytest
from py_base import Setup, Algorithm, Base, run

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
def ret():            return "Ret"

def _load(d, kind, ty, addr, off, trusted=False):
    return {"Load": [d, {"kind": kind, "ty": ty, "notrap_aligned": trusted}, addr, off]}

def load64(d, addr, off=0): return _load(d, "Plain", "I64", addr, off)
def load32(d, addr, off=0): return _load(d, "Plain", "I32", addr, off)




# CLIF reads data_ptr (offset 0x08), data_len (offset 0x10), out_ptr (offset 0x18).
# Copies each i32 from data to out, multiplied by 2.

DOUBLE_I32_PROG = program([
    function(0, [
        block(0, [0], [
            ret(),
        ]),
    ]),
    function(1, [
        block(0, [0], [
            iconst64(1, 8),
            iadd(2, 0, 1),
            load64(3, 2),
            iconst64(4, 16),
            iadd(5, 0, 4),
            load64(6, 5),
            iconst64(7, 24),
            iadd(8, 0, 7),
            load64(9, 8),
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

# CLIF that writes values to shared memory for Arrow output.
# Writes to fixed offsets above the 0x28 reserved region.
# Memory layout:
#   0x28: row_count (i64)
#   0x30: i64 column data (row_count * 8 bytes)
#   0x130: f64 column data (row_count * 8 bytes)
#   0x230: utf8 data (null-terminated strings)
#   0x330: utf8 byte length (i64)
ARROW_PROG = program([
    function(0, [
        block(0, [0], [
            ret(),
        ]),
    ]),
    function(1, [
        block(0, [0], [
            # row_count = 3
            iconst64(1, 3),
            iconst64(2, 40),
            iadd(3, 0, 2),
            store(1, 3),
            # i64 column: [100, 200, 300] at offset 0x30
            iconst64(4, 100),
            iconst64(5, 48),
            iadd(6, 0, 5),
            store(4, 6),
            iconst64(7, 200),
            iconst64(8, 56),
            iadd(9, 0, 8),
            store(7, 9),
            iconst64(10, 300),
            iconst64(11, 64),
            iadd(12, 0, 11),
            store(10, 12),
            # f64 column: [1.5, 2.5, 3.5] at offset 0x130
            # 1.5 = 0x3FF8000000000000
            iconst64(13, 4609434218613702656),
            iconst64(14, 304),
            iadd(15, 0, 14),
            store(13, 15),
            # 2.5 = 0x4004000000000000
            iconst64(16, 4612811918334230528),
            iconst64(17, 312),
            iadd(18, 0, 17),
            store(16, 18),
            # 3.5 = 0x400C000000000000
            iconst64(19, 4615063718147915776),
            iconst64(20, 320),
            iadd(21, 0, 20),
            store(19, 21),
            # utf8 column: "hello\0world\0foo\0" at offset 0x230
            # "hello" = 68 65 6c 6c 6f 00
            iconst64(22, 560),
            iadd(23, 0, 22),
            # 'h'=104 'e'=101 'l'=108 'l'=108 'o'=111 0
            iconst64(24, 478560413032),
            store(24, 23),
            # "world" = 77 6f 72 6c 64 00
            iconst64(25, 566),
            iadd(26, 0, 25),
            iconst64(27, 431316168567),
            store(27, 26),
            # "foo" = 66 6f 6f 00
            iconst64(28, 572),
            iadd(29, 0, 28),
            iconst64(30, 7303014),
            store(30, 29),
            # utf8 total byte length = 18 at offset 0x330
            iconst64(31, 18),
            iconst64(32, 816),
            iadd(33, 0, 32),
            store(31, 33),
            ret(),
        ]),
    ]),
])


COMPACT_IO_OFFSETS = {
    "data_ptr": 8,
    "data_len": 16,
    "out_ptr": 24,
    "out_len": 32,
}


def make_double_config():
    return json.dumps({
        "clif": DOUBLE_I32_PROG,
        "memory_size": 256,
        "io_offsets": COMPACT_IO_OFFSETS,
        "initial_memory": [0] * 256,
    })


def make_algorithm():
    return json.dumps({"fn_idx": 1, "output": []})


def make_arrow_config():
    return json.dumps({
        "clif": ARROW_PROG,
        "memory_size": 1024,
        "io_offsets": COMPACT_IO_OFFSETS,
        "initial_memory": [0] * 1024,
    })


def make_arrow_algorithm_i64():
    """Single i64 column output."""
    return json.dumps({
        "fn_idx": 1,
        "output": [{
            "columns": [{"name": "ids", "dtype": "I64", "data_offset": 48, "len_offset": 0}],
            "row_count_offset": 40,
        }],
    })


def make_arrow_algorithm_f64():
    """Single f64 column output."""
    return json.dumps({
        "fn_idx": 1,
        "output": [{
            "columns": [{"name": "scores", "dtype": "F64", "data_offset": 304, "len_offset": 0}],
            "row_count_offset": 40,
        }],
    })


def make_arrow_algorithm_utf8():
    """Single utf8 column output."""
    return json.dumps({
        "fn_idx": 1,
        "output": [{
            "columns": [{"name": "names", "dtype": "Utf8", "data_offset": 560, "len_offset": 816}],
            "row_count_offset": 40,
        }],
    })


def make_arrow_algorithm_multi_column():
    """Multiple columns (i64 + f64) in one batch."""
    return json.dumps({
        "fn_idx": 1,
        "output": [{
            "columns": [
                {"name": "ids", "dtype": "I64", "data_offset": 48, "len_offset": 0},
                {"name": "scores", "dtype": "F64", "data_offset": 304, "len_offset": 0},
            ],
            "row_count_offset": 40,
        }],
    })


def make_arrow_algorithm_multi_batch():
    """Two separate batches from the same execution."""
    return json.dumps({
        "fn_idx": 1,
        "output": [
            {
                "columns": [{"name": "ids", "dtype": "I64", "data_offset": 48, "len_offset": 0}],
                "row_count_offset": 40,
            },
            {
                "columns": [{"name": "scores", "dtype": "F64", "data_offset": 304, "len_offset": 0}],
                "row_count_offset": 40,
            },
        ],
    })


def pack_i32s(values):
    return struct.pack(f"<{len(values)}i", *values)


def unpack_i32s(data, count):
    return list(struct.unpack(f"<{count}i", data[:count * 4]))


class TestSetup:
    def test_valid_json(self):
        config = Setup(make_double_config())
        assert config is not None

    def test_invalid_json(self):
        with pytest.raises(ValueError, match="Invalid Setup JSON"):
            Setup("not json")

    def test_missing_fields(self):
        with pytest.raises(ValueError):
            Setup('{"clif": {"functions": []}}')


class TestAlgorithm:
    def test_valid_json(self):
        alg = Algorithm(make_algorithm())
        assert alg is not None

    def test_invalid_json(self):
        with pytest.raises(ValueError, match="Invalid Algorithm JSON"):
            Algorithm("{bad")

    def test_reuse(self):
        alg = Algorithm(make_algorithm())
        ref1 = alg
        ref2 = alg
        assert ref1 is ref2


class TestBase:
    def test_new(self):
        config = Setup(make_double_config())
        base = Base(config)
        assert base is not None

    def test_malformed_program(self):
        """v9 is never defined, so the program cannot be built."""
        config_json = json.dumps({
            "clif": program([function(0, [block(0, [0], [store(9, 0), ret()])])]),
            "memory_size": 256,
            "io_offsets": COMPACT_IO_OFFSETS,
            "initial_memory": [0] * 256,
        })
        with pytest.raises(ValueError, match="Base::new failed"):
            Base(Setup(config_json))

    def test_execute_returns_list(self):
        config = Setup(make_double_config())
        alg = Algorithm(make_algorithm())
        base = Base(config)
        result = base.execute(alg)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_execute_no_data(self):
        config = Setup(make_double_config())
        alg = Algorithm(make_algorithm())
        base = Base(config)
        result = base.execute(alg)
        assert isinstance(result, list)

    def test_execute_into_doubles(self):
        config = Setup(make_double_config())
        alg = Algorithm(make_algorithm())
        base = Base(config)

        values = [1, 2, 3, 4, 5, 10, 100, -7]
        data = pack_i32s(values)
        out = bytearray(len(data))
        base.execute_into(alg, data, out)

        result = unpack_i32s(out, len(values))
        assert result == [v * 2 for v in values]

    def test_execute_into_reuse(self):
        config = Setup(make_double_config())
        alg = Algorithm(make_algorithm())
        base = Base(config)

        for seed in range(5):
            values = list(range(seed * 10, seed * 10 + 20))
            data = pack_i32s(values)
            out = bytearray(len(data))
            base.execute_into(alg, data, out)
            result = unpack_i32s(out, len(values))
            assert result == [v * 2 for v in values]

    def test_execute_into_large(self):
        config = Setup(make_double_config())
        alg = Algorithm(make_algorithm())
        base = Base(config)

        n = 100_000
        values = list(range(n))
        data = pack_i32s(values)
        out = bytearray(len(data))
        base.execute_into(alg, data, out)

        result = unpack_i32s(out, n)
        for i in range(n):
            assert result[i] == values[i] * 2, f"Mismatch at index {i}"

    def test_execute_into_empty(self):
        config = Setup(make_double_config())
        alg = Algorithm(make_algorithm())
        base = Base(config)
        out = bytearray(0)
        base.execute_into(alg, b"", out)

    def test_bytes_input(self):
        config = Setup(make_double_config())
        alg = Algorithm(make_algorithm())
        base = Base(config)

        data = bytes(pack_i32s([42, -1, 0]))
        out = bytearray(len(data))
        base.execute_into(alg, data, out)
        assert unpack_i32s(out, 3) == [84, -2, 0]

    def test_execute_into_returns_list(self):
        config = Setup(make_double_config())
        alg = Algorithm(make_algorithm())
        base = Base(config)

        out = bytearray(16)
        result = base.execute_into(alg, pack_i32s([1, 2, 3, 4]), out)
        assert isinstance(result, list)
        assert len(result) == 0


class TestRun:
    def test_oneshot(self):
        config = Setup(make_double_config())
        alg = Algorithm(make_algorithm())
        result = run(config, alg)
        assert isinstance(result, list)



pa = pytest.importorskip("pyarrow")


class TestArrowI64:
    def test_single_i64_column(self):
        config = Setup(make_arrow_config())
        alg = Algorithm(make_arrow_algorithm_i64())
        base = Base(config)

        result = base.execute(alg)
        assert len(result) == 1
        batch = result[0]
        assert isinstance(batch, pa.RecordBatch)
        assert batch.num_rows == 3
        assert batch.num_columns == 1
        assert batch.column_names == ["ids"]
        assert batch.column("ids").to_pylist() == [100, 200, 300]

    def test_i64_column_types(self):
        config = Setup(make_arrow_config())
        alg = Algorithm(make_arrow_algorithm_i64())
        base = Base(config)

        batch = base.execute(alg)[0]
        assert batch.schema.field("ids").type == pa.int64()

    def test_i64_reuse_across_executes(self):
        config = Setup(make_arrow_config())
        alg = Algorithm(make_arrow_algorithm_i64())
        base = Base(config)

        batch1 = base.execute(alg)[0]
        batch2 = base.execute(alg)[0]
        assert batch1.column("ids").to_pylist() == [100, 200, 300]
        assert batch2.column("ids").to_pylist() == [100, 200, 300]


class TestArrowF64:
    def test_single_f64_column(self):
        config = Setup(make_arrow_config())
        alg = Algorithm(make_arrow_algorithm_f64())
        base = Base(config)

        batch = base.execute(alg)[0]
        assert batch.num_rows == 3
        assert batch.column_names == ["scores"]
        values = batch.column("scores").to_pylist()
        assert abs(values[0] - 1.5) < 1e-10
        assert abs(values[1] - 2.5) < 1e-10
        assert abs(values[2] - 3.5) < 1e-10

    def test_f64_column_type(self):
        config = Setup(make_arrow_config())
        alg = Algorithm(make_arrow_algorithm_f64())
        base = Base(config)

        batch = base.execute(alg)[0]
        assert batch.schema.field("scores").type == pa.float64()


class TestArrowUtf8:
    def test_single_utf8_column(self):
        config = Setup(make_arrow_config())
        alg = Algorithm(make_arrow_algorithm_utf8())
        base = Base(config)

        batch = base.execute(alg)[0]
        assert batch.num_rows == 3
        assert batch.column_names == ["names"]
        assert batch.column("names").to_pylist() == ["hello", "world", "foo"]

    def test_utf8_column_type(self):
        config = Setup(make_arrow_config())
        alg = Algorithm(make_arrow_algorithm_utf8())
        base = Base(config)

        batch = base.execute(alg)[0]
        assert batch.schema.field("names").type == pa.string()


class TestArrowMultiColumn:
    def test_two_columns(self):
        config = Setup(make_arrow_config())
        alg = Algorithm(make_arrow_algorithm_multi_column())
        base = Base(config)

        batch = base.execute(alg)[0]
        assert batch.num_rows == 3
        assert batch.num_columns == 2
        assert batch.column_names == ["ids", "scores"]
        assert batch.column("ids").to_pylist() == [100, 200, 300]
        scores = batch.column("scores").to_pylist()
        assert abs(scores[0] - 1.5) < 1e-10
        assert abs(scores[1] - 2.5) < 1e-10
        assert abs(scores[2] - 3.5) < 1e-10

    def test_multi_column_schema(self):
        config = Setup(make_arrow_config())
        alg = Algorithm(make_arrow_algorithm_multi_column())
        base = Base(config)

        batch = base.execute(alg)[0]
        assert batch.schema.field("ids").type == pa.int64()
        assert batch.schema.field("scores").type == pa.float64()


class TestArrowMultiBatch:
    def test_two_batches(self):
        config = Setup(make_arrow_config())
        alg = Algorithm(make_arrow_algorithm_multi_batch())
        base = Base(config)

        result = base.execute(alg)
        assert len(result) == 2

        assert result[0].column_names == ["ids"]
        assert result[0].column("ids").to_pylist() == [100, 200, 300]

        assert result[1].column_names == ["scores"]
        scores = result[1].column("scores").to_pylist()
        assert abs(scores[0] - 1.5) < 1e-10


class TestArrowWithExecuteInto:
    def test_arrow_and_bytearray_together(self):
        """execute_into can return Arrow batches AND write to bytearray."""
        config = Setup(make_arrow_config())
        alg = Algorithm(make_arrow_algorithm_i64())
        base = Base(config)

        out = bytearray(64)
        result = base.execute_into(alg, b"", out)
        assert len(result) == 1
        assert result[0].column("ids").to_pylist() == [100, 200, 300]


class TestArrowEmpty:
    def test_no_output_schema(self):
        config = Setup(make_double_config())
        alg = Algorithm(make_algorithm())
        base = Base(config)

        result = base.execute(alg)
        assert result == []

    def test_zero_rows(self):
        """If CLIF writes row_count=0, no batch is returned."""
        # The default initial_memory is all zeros, so row_count at offset 40 = 0.
        # We use a noop program that doesn't write anything.
        noop = program([
            function(0, [block(0, [0], [ret()])]),
            function(1, [block(0, [0], [ret()])]),
        ])
        config = Setup(json.dumps({
            "clif": noop,
            "memory_size": 256,
            "io_offsets": COMPACT_IO_OFFSETS,
            "initial_memory": [0] * 256,
        }))
        alg = Algorithm(json.dumps({
            "fn_idx": 1,
            "output": [{
                "columns": [{"name": "x", "dtype": "I64", "data_offset": 48, "len_offset": 0}],
                "row_count_offset": 40,
            }],
        }))
        base = Base(config)
        result = base.execute(alg)
        assert result == []


class TestArrowRunOneshot:
    def test_run_returns_arrow(self):
        config = Setup(make_arrow_config())
        alg = Algorithm(make_arrow_algorithm_i64())
        result = run(config, alg)
        assert len(result) == 1
        assert result[0].column("ids").to_pylist() == [100, 200, 300]
