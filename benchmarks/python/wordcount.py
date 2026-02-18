"""Word frequency count benchmark — pure Python.

Usage: python3 wordcount.py <text_file>

Reads a text file, counts word frequencies, prints "word\\tcount" sorted.
"""
import sys

path = sys.argv[1]
counts = {}
with open(path, "r") as f:
    for line in f:
        for word in line.split():
            counts[word] = counts.get(word, 0) + 1

for word, count in counts.items():
    print(f"{word}\t{count}")
