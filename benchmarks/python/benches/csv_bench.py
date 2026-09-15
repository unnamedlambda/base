import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness
import py_base
import entries


SIZES = [10_000, 100_000, 500_000, 1_000_000, 2_000_000]


def generate_csv(path: str, n: int) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    total = 0
    with open(path, "w") as f:
        f.write("id,first_name,last_name,email,department,salary\n")
        for i in range(n):
            salary = 1000 + ((i * 137) % 9000)
            total += salary
            f.write(f"{i},First{i},Last{i},e{i}@co.com,Dept{i % 10},{salary}\n")
    return total


def python_csv_sum(path: str, output_path: str) -> int:
    total = 0
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = header.index("salary")
        for row in reader:
            total += int(row[idx])
    harness.write_result_sync(output_path, total)
    return total


def build_payload(csv_path: str, output_path: str) -> bytes:
    return (csv_path + "\x00" + output_path + "\x00").encode()


def run(algo_path: str, rounds: int) -> list[harness.BenchResult]:
    artifact = py_base.load_artifact(algo_path)
    artifact_entries = entries.entries(os.path.basename(algo_path).removesuffix(".json"))
    engine = py_base.Base(artifact)
    alg = artifact_entries["main"]
    results = []

    for n in SIZES:
        csv_path = f"/tmp/bench-data/employees_{n}.csv"
        output_path = f"/tmp/bench-data/py_csv_result_{n}.txt"
        payload = build_payload(csv_path, output_path)

        expected = generate_csv(csv_path, n)

        py_out = output_path + ".py"

        python_ms = harness.median_of(rounds, lambda: harness.time_ms(
            lambda: python_csv_sum(csv_path, py_out)
        ))

        # Warmup
        if os.path.exists(output_path):
            os.remove(output_path)
        engine.execute(alg, payload)

        # The warmup created the output file, so each timed run overwrites a
        # path that exists — the same filesystem work the baseline does. A
        # remove here would charge Base for an inode creation a round instead.
        def run_pybase():
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
            name=f"CSV ({harness.format_count(n)})",
            python_ms=python_ms,
            pybase_ms=pybase_ms,
            verified=verified,
        ))

    return results
