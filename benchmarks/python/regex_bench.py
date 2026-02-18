"""Regex benchmark — pure Python.

Usage: python3 regex_bench.py <text_file>

Counts how many words match the pattern [a-z]+ing (words ending in 'ing').
Prints the count.
"""
import re
import sys

path = sys.argv[1]
pattern = re.compile(r'\b[a-z]+ing\b')
with open(path, "r") as f:
    data = f.read()
count = len(pattern.findall(data))
print(count)
