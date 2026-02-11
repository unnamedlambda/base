"""CSV benchmark — pure Python (stdlib only).

Usage: python3 csv_pure.py <csv_file>

Reads a CSV file with a 'salary' column and prints the sum.
"""

import csv
import sys

path = sys.argv[1]
total = 0
with open(path, "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    salary_idx = header.index("salary")
    for row in reader:
        total += int(row[salary_idx])

print(total)
