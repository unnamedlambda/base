"""The numbers every double parser gets wrong somewhere, as one JSON array.

    python3 numbers.py <out.json>

Exact halfway cases at 1, 2^53 and the largest double; the smallest
subnormal and its neighbours; long fractions with leading zeros; then 20000
numbers of 20 to 80 digits across the doubles' range and 20000 subnormals,
from a fixed seed. `run.sh` checks base's checksum over them against
simdjson's before anything is timed: every double must be the nearest to its
text, bit for bit.
"""

import random
import sys

FIXED = [
    "1.00000000000000011102230246251565404236316680908203125",
    "1.00000000000000011102230246251565404236316680908203126",
    "1.00000000000000011102230246251565404236316680908203124",
    "9007199254740993.0", "9007199254740995.0", "9007199254740993.000000000000000000001",
    "2.2250738585072011e-308", "2.2250738585072012e-308", "2.2250738585072014e-308",
    "4.9e-324", "5e-324", "2.4703282292062328e-324", "2.4703282292062327e-324", "1e-320",
    "0.0000000000000000000000000000000000000000000000000000000000000001",
    "-0.000000000000000000000123456789012345678901",
    "1.7976931348623157e308", "1.7976931348623158e308", "3.141592653589793238462643383279502884197",
    "1e-400", "0.1e-342", "7.2057594037927933e16", "8.98846567431158e307",
    "0.30000000000000000555111512312578270211815834045410156250",
    "1.4999999999999999999999999999999999999999e-300",
]


def main():
    rng = random.Random(7)
    vals = list(FIXED)
    for _ in range(20000):
        nd = rng.randint(20, 80)
        digs = str(rng.randint(1, 9)) + "".join(rng.choice("0123456789") for _ in range(nd - 1))
        pt = rng.randint(1, nd - 1)
        # about 10^(pt + e): kept within the doubles, which simdjson requires
        e = rng.randint(-330 - pt, 307 - pt)
        vals.append(f"{'-' if rng.random() < 0.5 else ''}{digs[:pt]}.{digs[pt:]}e{e}")
    for _ in range(20000):
        d = "".join(rng.choice("0123456789") for _ in range(rng.randint(1, 40)))
        vals.append(f"{rng.randint(1, 9)}.{d}e-{rng.randint(308, 323)}")
    with open(sys.argv[1], "w") as f:
        f.write("[" + ",".join(vals) + "]")


if __name__ == "__main__":
    main()
