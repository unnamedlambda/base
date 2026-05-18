import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness
import py_base


SIZES = [1_000, 10_000, 50_000, 100_000, 500_000]


def generate_json(path: str, n: int) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    total = 0
    with open(path, "w") as f:
        f.write("[\n")
        for i in range(n):
            value = ((i * 137 + 42) % 10000)
            total += value
            sep = ",\n" if i > 0 else ""
            f.write(f'{sep}  {{"id": {i}, "name": "item_{i}", "value": {value}}}')
        f.write("\n]\n")
    return total


def python_json_sum(path: str) -> int:
    with open(path) as f:
        data = json.load(f)
    return sum(item["value"] for item in data)


def build_payload(json_path: str, output_path: str) -> bytes:
    return (json_path + "\x00" + output_path + "\x00").encode()


def run(algo_path: str, rounds: int) -> list[harness.BenchResult]:
    engine, alg = py_base.load(algo_path)
    results = []

    for n in SIZES:
        json_path = f"/tmp/bench-data/data_{n}.json"
        output_path = f"/tmp/bench-data/py_json_result_{n}.txt"
        payload = build_payload(json_path, output_path)

        expected = generate_json(json_path, n)

        python_ms = harness.median_of(rounds, lambda: harness.time_ms(
            lambda: python_json_sum(json_path)
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
            name=f"JSON ({harness.format_count(n)})",
            python_ms=python_ms,
            pybase_ms=pybase_ms,
            verified=verified,
        ))

    return results
