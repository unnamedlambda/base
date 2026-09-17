"""CBOR for the tests, standard library only.

`encode` writes the profile `base_types::Artifact` reads, so a test can build an
artifact as nested dicts: a dict is a map whose keys stay in the order written,
`bytes` is a byte string, `None` is null. `decode` reads any definite-length
CBOR a program answers with, including typed arrays (RFC 8746), which it
returns as `(tag, bytes)` for the caller to interpret.
"""

import struct


def _head(major, n):
    if n < 24:
        return bytes([major << 5 | n])
    for info, fmt, limit in ((24, ">B", 0xFF), (25, ">H", 0xFFFF),
                             (26, ">I", 0xFFFFFFFF), (27, ">Q", 2**64 - 1)):
        if n <= limit:
            return bytes([major << 5 | info]) + struct.pack(fmt, n)
    raise ValueError(f"{n} does not fit a CBOR head")


def encode(v):
    if v is None:
        return b"\xf6"
    if v is True:
        return b"\xf5"
    if v is False:
        return b"\xf4"
    if isinstance(v, int):
        return _head(0, v) if v >= 0 else _head(1, -1 - v)
    if isinstance(v, (bytes, bytearray)):
        return _head(2, len(v)) + bytes(v)
    if isinstance(v, str):
        b = v.encode()
        return _head(3, len(b)) + b
    if isinstance(v, (list, tuple)):
        return _head(4, len(v)) + b"".join(map(encode, v))
    if isinstance(v, dict):
        return _head(5, len(v)) + b"".join(encode(k) + encode(x) for k, x in v.items())
    raise TypeError(f"cannot encode {type(v).__name__}")


def decode(data):
    """The one value `data` holds; trailing bytes are an error."""
    value, end = _item(memoryview(data), 0)
    if end != len(data):
        raise ValueError(f"{len(data) - end} bytes follow the value")
    return value


def _item(buf, i):
    initial = buf[i]
    major, info = initial >> 5, initial & 0x1F
    i += 1
    if major == 7:
        simple = {20: False, 21: True, 22: None}
        if info in simple:
            return simple[info], i
        if info in (25, 26, 27):
            size, fmt = {25: (2, ">e"), 26: (4, ">f"), 27: (8, ">d")}[info]
            return struct.unpack(fmt, buf[i:i + size])[0], i + size
        raise ValueError(f"unsupported simple value {info}")
    if info < 24:
        n = info
    elif info <= 27:
        size = 1 << (info - 24)
        n = int.from_bytes(buf[i:i + size], "big")
        i += size
    else:
        raise ValueError("indefinite lengths are not read here")
    if major == 0:
        return n, i
    if major == 1:
        return -1 - n, i
    if major == 2:
        return bytes(buf[i:i + n]), i + n
    if major == 3:
        return str(buf[i:i + n], "utf-8"), i + n
    if major == 4:
        out = []
        for _ in range(n):
            x, i = _item(buf, i)
            out.append(x)
        return out, i
    if major == 5:
        out = {}
        for _ in range(n):
            k, i = _item(buf, i)
            out[k], i = _item(buf, i)
        return out, i
    x, i = _item(buf, i)
    return (n, x), i
