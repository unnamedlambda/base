"""String search benchmark — pure Python.

Usage: python3 string_search.py <text_file> <pattern>

Counts occurrences of a pattern in the text file.
Prints the count.
"""
import sys

path = sys.argv[1]
pattern = sys.argv[2]
with open(path, "r") as f:
    data = f.read()
count = 0
start = 0
while True:
    pos = data.find(pattern, start)
    if pos == -1:
        break
    count += 1
    start = pos + 1
print(count)
