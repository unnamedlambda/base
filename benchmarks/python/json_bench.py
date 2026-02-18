"""JSON benchmark — pure Python.

Usage: python3 json_bench.py <json_file>

Reads a JSON file containing an array of objects, each with a "value" field.
Prints the sum of all "value" fields.
"""
import json
import sys

path = sys.argv[1]
with open(path, "r") as f:
    data = json.load(f)
total = sum(item["value"] for item in data)
print(total)
