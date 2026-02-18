"""Sort benchmark — pure Python.

Usage: python3 sort_bench.py <binary_file>

Reads a binary file of little-endian i32 values, sorts them,
prints the first and last value as "first,last".
"""
import struct
import sys

path = sys.argv[1]
with open(path, "rb") as f:
    data = f.read()
n = len(data) // 4
values = list(struct.unpack(f"<{n}i", data))
values.sort()
print(f"{values[0]},{values[-1]}")
