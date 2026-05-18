import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness
import py_base


SIZES = [10_000, 50_000, 100_000, 500_000, 1_000_000]

VOCABULARY = [
    "the", "of", "and", "to", "in", "a", "is", "that",
    "for", "it", "was", "on", "are", "as", "with", "his",
    "they", "at", "be", "this", "from", "or", "had", "by",
    "not", "but", "some", "what", "we", "can", "out", "all",
    "your", "when", "up", "use", "how", "said", "an", "each",
]

PATTERN = "that"


def generate_text(path: str, n: int) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    words = []
    for i in range(n):
        idx = ((i * 7 + 13) * 31) % len(VOCABULARY)
        words.append(VOCABULARY[idx])
    text = " ".join(words)
    with open(path, "w") as f:
        f.write(text + "\n")
    # Count overlapping occurrences
    count = 0
    start = 0
    while True:
        pos = text.find(PATTERN, start)
        if pos == -1:
            break
        count += 1
        start = pos + 1
    return count


def python_strsearch(path: str) -> int:
    with open(path) as f:
        text = f.read()
    count = 0
    start = 0
    while True:
        pos = text.find(PATTERN, start)
        if pos == -1:
            break
        count += 1
        start = pos + 1
    return count


def build_payload(text_path: str, output_path: str) -> bytes:
    return (text_path + "\x00" + output_path + "\x00").encode()


def run(algo_path: str, rounds: int) -> list[harness.BenchResult]:
    engine, alg = py_base.load(algo_path)
    results = []

    for n in SIZES:
        text_path = f"/tmp/bench-data/strsearch_{n}.txt"
        output_path = f"/tmp/bench-data/py_strsearch_result_{n}.txt"
        payload = build_payload(text_path, output_path)

        expected = generate_text(text_path, n)

        python_ms = harness.median_of(rounds, lambda: harness.time_ms(
            lambda: python_strsearch(text_path)
        ))

        # Warmup
        if os.path.exists(output_path):
            os.remove(output_path)
        engine.execute(alg, payload)

        def run_pybase():
            if os.path.exists(output_path):
                os.remove(output_path)
            return harness.time_ms(lambda: engine.execute(alg, payload))

        pybase_ms = harness.median_of(rounds, run_pybase)

        verified = None
        if os.path.exists(output_path):
            with open(output_path) as f:
                try:
                    verified = int(f.read().strip()) == expected
                except ValueError:
                    verified = False

        results.append(harness.BenchResult(
            name=f"StrSearch ({harness.format_count(n)})",
            python_ms=python_ms,
            pybase_ms=pybase_ms,
            verified=verified,
        ))

    return results
