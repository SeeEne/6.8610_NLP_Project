#!/usr/bin/env python3
"""Normalize DS-1000 tasks from harness format to concatenation-friendly format.

Reads from data/raw/ds1000.jsonl, writes normalized tasks to data/raw/ds1000_normalized.jsonl.

Usage:
    python scripts/normalize_ds1000.py
    python scripts/normalize_ds1000.py --verify 5    # verify N tasks in Docker sandbox
    python scripts/normalize_ds1000.py --output data/raw/ds1000_custom.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.model import BenchmarkTask
from src.data.ds1000_normalizer import normalize_all


def load_tasks(path: Path) -> list[BenchmarkTask]:
    tasks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(BenchmarkTask.from_dict(json.loads(line)))
    return tasks


def save_tasks(tasks: list[BenchmarkTask], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for task in tasks:
            f.write(json.dumps(task.to_dict(), ensure_ascii=False) + "\n")


def verify_in_sandbox(tasks: list[BenchmarkTask], n: int) -> None:
    """Verify N normalized tasks execute correctly in Docker sandbox."""
    from src.util.sandbox import Sandbox

    print(f"\nVerifying {n} tasks in Docker sandbox...")
    sandbox = Sandbox(image="ambicode-ds1000", timeout_s=60)

    # Pick tasks spread across libraries
    by_lib = {}
    for t in tasks:
        lib = t.library or "unknown"
        by_lib.setdefault(lib, []).append(t)

    selected = []
    for lib in sorted(by_lib.keys()):
        per_lib = max(1, n // len(by_lib))
        selected.extend(by_lib[lib][:per_lib])
    selected = selected[:n]

    passed = 0
    failed = 0
    for task in selected:
        result = sandbox.run(task.canonical_solution, task.test_code, timeout_s=60)
        status = "PASS" if result.passed else "FAIL"
        if result.passed:
            passed += 1
        else:
            failed += 1
            stderr_line = result.stderr.strip().split("\n")[-1][:120] if result.stderr else ""
            print(f"  {status}  {task.task_id:30s}  {stderr_line}")
        if result.passed:
            print(f"  {status}  {task.task_id:30s}")

    print(f"\n  Verified: {passed}/{passed + failed} passed")


def main():
    parser = argparse.ArgumentParser(description="Normalize DS-1000 tasks")
    parser.add_argument(
        "--input",
        default="data/raw/ds1000.jsonl",
        help="Input JSONL path (default: data/raw/ds1000.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="data/raw/ds1000_normalized.jsonl",
        help="Output JSONL path (default: data/raw/ds1000_normalized.jsonl)",
    )
    parser.add_argument(
        "--verify",
        type=int,
        default=0,
        metavar="N",
        help="Verify N tasks in Docker sandbox after normalization",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: {input_path} not found. Run scripts/download_data.py first.")
        sys.exit(1)

    print(f"Loading tasks from {input_path}...")
    tasks = load_tasks(input_path)
    print(f"  Loaded {len(tasks)} tasks")

    print(f"\nNormalizing...")
    normalized, stats = normalize_all(tasks)

    print(f"\n  Results:")
    print(f"    Total DS1000 tasks:  {stats['total']}")
    print(f"    Normalized:          {stats['success']}")
    print(f"    Skipped:             {stats['skipped']}")
    for reason, count in stats.get("skip_reasons", {}).items():
        print(f"      - {reason}: {count}")
    print(f"    Errors:              {stats['errors']}")
    for err in stats.get("error_details", []):
        print(f"      - {err}")

    save_tasks(normalized, output_path)
    print(f"\n  Saved {len(normalized)} tasks to {output_path}")

    if args.verify > 0:
        verify_in_sandbox(normalized, args.verify)


if __name__ == "__main__":
    main()
