import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness
import py_base


SIZES = [10_000, 50_000, 100_000, 500_000, 1_000_000]

VOCABULARY = [
    "the", "running", "of", "singing", "and", "to", "jumping", "in",
    "a", "is", "that", "finding", "for", "it", "was", "making",
    "on", "are", "as", "with", "building", "they", "at", "be",
    "this", "from", "or", "testing", "had", "by", "not", "coding",
    "but", "some", "what", "we", "writing", "can", "out", "reading",
]

def generate_text(path: str, n: int) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    words = []
    expected = 0
    for i in range(n):
        idx = ((i * 7 + 13) * 31) % len(VOCABULARY)
        word = VOCABULARY[idx]
        if word.endswith("ing"):
            expected += 1
        words.append(word)
    with open(path, "w") as f:
        f.write(" ".join(words) + "\n")
    return expected


def python_regex_count(path: str) -> int:
    with open(path) as f:
        text = f.read()
    return sum(1 for w in text.split() if len(w) > 3 and w.endswith("ing"))


def build_payload(text_path: str, output_path: str) -> bytes:
    return (text_path + "\x00" + output_path + "\x00").encode()


def run(algo_path: str, rounds: int) -> list[harness.BenchResult]:
    engine, alg = py_base.load(algo_path)
    results = []

    for n in SIZES:
        text_path = f"/tmp/bench-data/regex_{n}.txt"
        output_path = f"/tmp/bench-data/py_regex_result_{n}.txt"
        payload = build_payload(text_path, output_path)

        expected = generate_text(text_path, n)

        python_ms = harness.median_of(rounds, lambda: harness.time_ms(
            lambda: python_regex_count(text_path)
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
            name=f"Regex ({harness.format_count(n)})",
            python_ms=python_ms,
            pybase_ms=pybase_ms,
            verified=verified,
        ))

    return results
