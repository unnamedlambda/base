import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness
import py_base


SIZES = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]

VOCABULARY = [
    "the", "of", "and", "to", "in", "a", "is", "that", "for", "it", "was", "on",
    "are", "as", "with", "his", "they", "at", "be", "this", "from", "or", "had",
    "by", "not", "but", "some", "what", "we", "can", "out", "all", "your",
    "when", "up", "use", "how", "said", "an", "each",
]


def generate_text(path: str, num_words: int) -> dict[str, int]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    expected: dict[str, int] = {}
    with open(path, "w") as f:
        for i in range(num_words):
            idx = ((i * 7 + 13) * 31) % len(VOCABULARY)
            word = VOCABULARY[idx]
            expected[word] = expected.get(word, 0) + 1
            if i > 0:
                f.write(" ")
            f.write(word)
        f.write("\n")
    return expected


def python_wordcount(path: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    with open(path, "r") as f:
        for line in f:
            for word in line.split():
                counts[word] = counts.get(word, 0) + 1
    return counts


def build_payload(text_path: str, output_path: str) -> bytes:
    return (text_path + "\x00" + output_path + "\x00").encode()


def parse_output(content: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for line in content.splitlines():
        if "\t" in line:
            word, count_str = line.split("\t", 1)
            try:
                result[word] = int(count_str)
            except ValueError:
                pass
    return result


def run(algo_path: str, rounds: int) -> list[harness.BenchResult]:
    artifact = py_base.load_artifact(algo_path)
    engine = py_base.Base(artifact.config)
    alg = artifact.main
    results = []

    for n in SIZES:
        text_path = f"/tmp/bench-data/words_{n}.txt"
        output_path = f"/tmp/bench-data/py_wc_result_{n}.txt"
        payload = build_payload(text_path, output_path)

        expected = generate_text(text_path, n)

        python_ms = harness.median_of(rounds, lambda: harness.time_ms(
            lambda: python_wordcount(text_path)
        ))

        # Fresh engine per execute: HT state accumulates across execute() calls.
        def run_pybase():
            if os.path.exists(output_path):
                os.remove(output_path)
            art = py_base.load_artifact(algo_path)
            eng = py_base.Base(art.config)
            a = art.main
            return harness.time_ms(lambda: eng.execute(a, payload))

        # Warmup
        run_pybase()

        pybase_ms = harness.median_of(rounds, run_pybase)

        verified = None
        if os.path.exists(output_path):
            with open(output_path) as f:
                got = parse_output(f.read().strip())
            verified = got == expected

        results.append(harness.BenchResult(
            name=f"WC ({harness.format_count(n)})",
            python_ms=python_ms,
            pybase_ms=pybase_ms,
            verified=verified,
        ))

    return results
