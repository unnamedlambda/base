import json
import struct
import pytest
from py_base import BaseConfig, Algorithm, Base, run


# CLIF reads data_ptr (offset 0x08), data_len (offset 0x10), out_ptr (offset 0x18).
# Copies each i32 from data to out, multiplied by 2.

DOUBLE_I32_CLIF = """\
function u0:0(i64) system_v {
block0(v0: i64):
    return
}

function u0:1(i64) system_v {
block0(v0: i64):
    v1 = iconst.i64 8
    v2 = iadd v0, v1
    v3 = load.i64 v2
    v4 = iconst.i64 16
    v5 = iadd v0, v4
    v6 = load.i64 v5
    v7 = iconst.i64 24
    v8 = iadd v0, v7
    v9 = load.i64 v8
    v10 = iconst.i64 2
    v11 = ushr v6, v10
    v20 = iconst.i64 0
    jump block1(v20)
block1(v12: i64):
    v21 = icmp ult v12, v11
    brif v21, block2(v12), block3
block2(v13: i64):
    v22 = iconst.i64 2
    v23 = ishl v13, v22
    v24 = iadd v3, v23
    v25 = load.i32 v24
    v26 = iconst.i32 1
    v27 = ishl v25, v26
    v28 = iadd v9, v23
    store v27, v28
    v29 = iconst.i64 1
    v30 = iadd v13, v29
    jump block1(v30)
block3:
    return
}
"""

# CLIF that writes values to shared memory for Arrow output.
# Writes to fixed offsets above the 0x28 reserved region.
# Memory layout:
#   0x28: row_count (i64)
#   0x30: i64 column data (row_count * 8 bytes)
#   0x130: f64 column data (row_count * 8 bytes)
#   0x230: utf8 data (null-terminated strings)
#   0x330: utf8 byte length (i64)
ARROW_CLIF = """\
function u0:0(i64) system_v {
block0(v0: i64):
    return
}

function u0:1(i64) system_v {
block0(v0: i64):
    ; row_count = 3
    v1 = iconst.i64 3
    v2 = iconst.i64 40
    v3 = iadd v0, v2
    store v1, v3
    ; i64 column: [100, 200, 300] at offset 0x30
    v4 = iconst.i64 100
    v5 = iconst.i64 48
    v6 = iadd v0, v5
    store v4, v6
    v7 = iconst.i64 200
    v8 = iconst.i64 56
    v9 = iadd v0, v8
    store v7, v9
    v10 = iconst.i64 300
    v11 = iconst.i64 64
    v12 = iadd v0, v11
    store v10, v12
    ; f64 column: [1.5, 2.5, 3.5] at offset 0x130
    ; 1.5 = 0x3FF8000000000000
    v13 = iconst.i64 4609434218613702656
    v14 = iconst.i64 304
    v15 = iadd v0, v14
    store v13, v15
    ; 2.5 = 0x4004000000000000
    v16 = iconst.i64 4612811918334230528
    v17 = iconst.i64 312
    v18 = iadd v0, v17
    store v16, v18
    ; 3.5 = 0x400C000000000000
    v19 = iconst.i64 4615063718147915776
    v20 = iconst.i64 320
    v21 = iadd v0, v20
    store v19, v21
    ; utf8 column: "hello\0world\0foo\0" at offset 0x230
    ; "hello" = 68 65 6c 6c 6f 00
    v22 = iconst.i64 560
    v23 = iadd v0, v22
    ; 'h'=104 'e'=101 'l'=108 'l'=108 'o'=111 0
    v24 = iconst.i64 0x006f6c6c6568
    store v24, v23
    ; "world" = 77 6f 72 6c 64 00
    v25 = iconst.i64 566
    v26 = iadd v0, v25
    v27 = iconst.i64 0x00646c726f77
    store v27, v26
    ; "foo" = 66 6f 6f 00
    v28 = iconst.i64 572
    v29 = iadd v0, v28
    v30 = iconst.i64 0x006f6f66
    store v30, v29
    ; utf8 total byte length = 18 at offset 0x330
    v31 = iconst.i64 18
    v32 = iconst.i64 816
    v33 = iadd v0, v32
    store v31, v33
    return
}
"""


def make_double_config():
    return json.dumps({
        "cranelift_ir": DOUBLE_I32_CLIF,
        "memory_size": 256,
        "context_offset": 0,
        "initial_memory": [0] * 256,
    })


def make_algorithm():
    return json.dumps({
        "actions": [{"kind": "clif_call", "dst": 0, "src": 1, "offset": 0, "size": 0}],
        "cranelift_units": 0,
        "timeout_ms": 10000,
        "output": [],
    })


def make_arrow_config():
    return json.dumps({
        "cranelift_ir": ARROW_CLIF,
        "memory_size": 1024,
        "context_offset": 0,
        "initial_memory": [0] * 1024,
    })


def make_arrow_algorithm_i64():
    """Single i64 column output."""
    return json.dumps({
        "actions": [{"kind": "clif_call", "dst": 0, "src": 1, "offset": 0, "size": 0}],
        "cranelift_units": 0,
        "timeout_ms": 10000,
        "output": [{
            "columns": [{"name": "ids", "dtype": "I64", "data_offset": 48, "len_offset": 0}],
            "row_count_offset": 40,
        }],
    })


def make_arrow_algorithm_f64():
    """Single f64 column output."""
    return json.dumps({
        "actions": [{"kind": "clif_call", "dst": 0, "src": 1, "offset": 0, "size": 0}],
        "cranelift_units": 0,
        "timeout_ms": 10000,
        "output": [{
            "columns": [{"name": "scores", "dtype": "F64", "data_offset": 304, "len_offset": 0}],
            "row_count_offset": 40,
        }],
    })


def make_arrow_algorithm_utf8():
    """Single utf8 column output."""
    return json.dumps({
        "actions": [{"kind": "clif_call", "dst": 0, "src": 1, "offset": 0, "size": 0}],
        "cranelift_units": 0,
        "timeout_ms": 10000,
        "output": [{
            "columns": [{"name": "names", "dtype": "Utf8", "data_offset": 560, "len_offset": 816}],
            "row_count_offset": 40,
        }],
    })


def make_arrow_algorithm_multi_column():
    """Multiple columns (i64 + f64) in one batch."""
    return json.dumps({
        "actions": [{"kind": "clif_call", "dst": 0, "src": 1, "offset": 0, "size": 0}],
        "cranelift_units": 0,
        "timeout_ms": 10000,
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
        "actions": [{"kind": "clif_call", "dst": 0, "src": 1, "offset": 0, "size": 0}],
        "cranelift_units": 0,
        "timeout_ms": 10000,
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


class TestBaseConfig:
    def test_valid_json(self):
        config = BaseConfig(make_double_config())
        assert config is not None

    def test_invalid_json(self):
        with pytest.raises(ValueError, match="Invalid BaseConfig JSON"):
            BaseConfig("not json")

    def test_missing_fields(self):
        with pytest.raises(ValueError):
            BaseConfig('{"cranelift_ir": ""}')


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
        config = BaseConfig(make_double_config())
        base = Base(config)
        assert base is not None

    def test_invalid_clif(self):
        config_json = json.dumps({
            "cranelift_ir": "not valid clif",
            "memory_size": 256,
            "context_offset": 0,
            "initial_memory": [0] * 256,
        })
        with pytest.raises(ValueError, match="Base::new failed"):
            Base(BaseConfig(config_json))

    def test_execute_returns_list(self):
        config = BaseConfig(make_double_config())
        alg = Algorithm(make_algorithm())
        base = Base(config)
        result = base.execute(alg)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_execute_no_data(self):
        config = BaseConfig(make_double_config())
        alg = Algorithm(make_algorithm())
        base = Base(config)
        result = base.execute(alg)
        assert isinstance(result, list)

    def test_execute_into_doubles(self):
        config = BaseConfig(make_double_config())
        alg = Algorithm(make_algorithm())
        base = Base(config)

        values = [1, 2, 3, 4, 5, 10, 100, -7]
        data = pack_i32s(values)
        out = bytearray(len(data))
        base.execute_into(alg, data, out)

        result = unpack_i32s(out, len(values))
        assert result == [v * 2 for v in values]

    def test_execute_into_reuse(self):
        config = BaseConfig(make_double_config())
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
        config = BaseConfig(make_double_config())
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
        config = BaseConfig(make_double_config())
        alg = Algorithm(make_algorithm())
        base = Base(config)
        out = bytearray(0)
        base.execute_into(alg, b"", out)

    def test_bytes_input(self):
        config = BaseConfig(make_double_config())
        alg = Algorithm(make_algorithm())
        base = Base(config)

        data = bytes(pack_i32s([42, -1, 0]))
        out = bytearray(len(data))
        base.execute_into(alg, data, out)
        assert unpack_i32s(out, 3) == [84, -2, 0]

    def test_execute_into_returns_list(self):
        config = BaseConfig(make_double_config())
        alg = Algorithm(make_algorithm())
        base = Base(config)

        out = bytearray(16)
        result = base.execute_into(alg, pack_i32s([1, 2, 3, 4]), out)
        assert isinstance(result, list)
        assert len(result) == 0


class TestRun:
    def test_oneshot(self):
        config = BaseConfig(make_double_config())
        alg = Algorithm(make_algorithm())
        result = run(config, alg)
        assert isinstance(result, list)



pa = pytest.importorskip("pyarrow")


class TestArrowI64:
    def test_single_i64_column(self):
        config = BaseConfig(make_arrow_config())
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
        config = BaseConfig(make_arrow_config())
        alg = Algorithm(make_arrow_algorithm_i64())
        base = Base(config)

        batch = base.execute(alg)[0]
        assert batch.schema.field("ids").type == pa.int64()

    def test_i64_reuse_across_executes(self):
        config = BaseConfig(make_arrow_config())
        alg = Algorithm(make_arrow_algorithm_i64())
        base = Base(config)

        batch1 = base.execute(alg)[0]
        batch2 = base.execute(alg)[0]
        assert batch1.column("ids").to_pylist() == [100, 200, 300]
        assert batch2.column("ids").to_pylist() == [100, 200, 300]


class TestArrowF64:
    def test_single_f64_column(self):
        config = BaseConfig(make_arrow_config())
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
        config = BaseConfig(make_arrow_config())
        alg = Algorithm(make_arrow_algorithm_f64())
        base = Base(config)

        batch = base.execute(alg)[0]
        assert batch.schema.field("scores").type == pa.float64()


class TestArrowUtf8:
    def test_single_utf8_column(self):
        config = BaseConfig(make_arrow_config())
        alg = Algorithm(make_arrow_algorithm_utf8())
        base = Base(config)

        batch = base.execute(alg)[0]
        assert batch.num_rows == 3
        assert batch.column_names == ["names"]
        assert batch.column("names").to_pylist() == ["hello", "world", "foo"]

    def test_utf8_column_type(self):
        config = BaseConfig(make_arrow_config())
        alg = Algorithm(make_arrow_algorithm_utf8())
        base = Base(config)

        batch = base.execute(alg)[0]
        assert batch.schema.field("names").type == pa.string()


class TestArrowMultiColumn:
    def test_two_columns(self):
        config = BaseConfig(make_arrow_config())
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
        config = BaseConfig(make_arrow_config())
        alg = Algorithm(make_arrow_algorithm_multi_column())
        base = Base(config)

        batch = base.execute(alg)[0]
        assert batch.schema.field("ids").type == pa.int64()
        assert batch.schema.field("scores").type == pa.float64()


class TestArrowMultiBatch:
    def test_two_batches(self):
        config = BaseConfig(make_arrow_config())
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
        config = BaseConfig(make_arrow_config())
        alg = Algorithm(make_arrow_algorithm_i64())
        base = Base(config)

        out = bytearray(64)
        result = base.execute_into(alg, b"", out)
        assert len(result) == 1
        assert result[0].column("ids").to_pylist() == [100, 200, 300]


class TestArrowEmpty:
    def test_no_output_schema(self):
        config = BaseConfig(make_double_config())
        alg = Algorithm(make_algorithm())
        base = Base(config)

        result = base.execute(alg)
        assert result == []

    def test_zero_rows(self):
        """If CLIF writes row_count=0, no batch is returned."""
        # The default initial_memory is all zeros, so row_count at offset 40 = 0.
        # We use a noop CLIF that doesn't write anything.
        noop_clif = """\
function u0:0(i64) system_v {
block0(v0: i64):
    return
}

function u0:1(i64) system_v {
block0(v0: i64):
    return
}
"""
        config = BaseConfig(json.dumps({
            "cranelift_ir": noop_clif,
            "memory_size": 256,
            "context_offset": 0,
            "initial_memory": [0] * 256,
        }))
        alg = Algorithm(json.dumps({
            "actions": [{"kind": "clif_call", "dst": 0, "src": 1, "offset": 0, "size": 0}],
            "cranelift_units": 0,
            "timeout_ms": 10000,
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
        config = BaseConfig(make_arrow_config())
        alg = Algorithm(make_arrow_algorithm_i64())
        result = run(config, alg)
        assert len(result) == 1
        assert result[0].column("ids").to_pylist() == [100, 200, 300]
