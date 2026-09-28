"""A Python host reading the `self_describing` artifact with no copy of its formats.

The counterpart of `applications/self-describing/tests/formats.rs`: the same
entries, read with the generic decoder in `cbor.py` and one format id.
"""

import os
import struct

import pytest
from py_base import Driver, load_artifact

import cbor

ARTIFACT = os.path.join(
    os.path.dirname(__file__), "..", "..", "lean", ".lake", "build", "artifacts",
    "self_describing.cbor")

INPUT = b"abca"

pytestmark = pytest.mark.skipif(
    not os.path.exists(ARTIFACT), reason="the Lean artifacts have not been generated")


def call(base, entry, data=b""):
    """Ask for the size, then call with a buffer that holds it."""
    need = base.execute(entry, data)
    out = bytearray(need)
    assert base.execute(entry, data, out) == need
    return bytes(out)


def histogram():
    counts = [0] * 256
    counts[ord("a")] = 2
    counts[ord("b")] = 1
    counts[ord("c")] = 1
    return struct.pack("<256Q", *counts)


@pytest.fixture
def base():
    return Driver(load_artifact(ARTIFACT))


def test_schema_describes_both_outputs(base):
    assert cbor.decode(call(base, "schema")) == {
        "stats": {
            "encoding": "cbor",
            "format": "base.u8stats/1",
            "fields": {
                "format": "text",
                "count": "uint",
                "sum": "uint",
                "histogram": "uint64le[256], tag 71",
            },
        },
        "bulk": {
            "encoding": "raw",
            "format_id": b"u8hist01",
            "layout": ["format_id", "uint64le[256]"],
        },
    }


def test_stats_decode_without_a_layout(base):
    assert cbor.decode(call(base, "stats", INPUT)) == {
        "format": "base.u8stats/1",
        "count": 4,
        "sum": 97 + 98 + 99 + 97,
        "histogram": (71, histogram()),
    }


def read_bulk(data, format_id):
    if data[:8] != format_id:
        raise ValueError(f"expected format {format_id!r}, got {data[:8]!r}")
    return list(struct.unpack("<256Q", data[8:]))


def test_bulk_is_read_after_checking_its_id(base):
    data = call(base, "bulk", INPUT)
    assert len(data) == 8 + 2048
    assert read_bulk(data, b"u8hist01") == list(struct.unpack("<256Q", histogram()))


def test_a_stale_bulk_reader_refuses(base):
    with pytest.raises(ValueError, match="expected format b'u8hist00', got b'u8hist01'"):
        read_bulk(call(base, "bulk", INPUT), b"u8hist00")
