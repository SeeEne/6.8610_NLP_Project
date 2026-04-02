#!/usr/bin/env python3
"""Run anchor selection pipeline on raw benchmark tasks.

Usage:
    python scripts/run_anchor_selection.py                           # all sources
    python scripts/run_anchor_selection.py --source humaneval        # one source
    python scripts/run_anchor_selection.py --source humaneval --limit 5   # first 5
    python scripts/run_anchor_selection.py --task-ids HumanEval/0 HumanEval/15  # specific tasks
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.store import BenchmarkStore
from src.pipeline.anchor_selection import run_anchor_selection
from src.pipeline.prompts import load_pipeline_config


def main():
    parser = argparse.ArgumentParser(description="Run anchor selection pipeline")
    parser.add_argument(
        "--source", type=str, help="Filter by source (humaneval, mbpp, ds1000)"
    )
    parser.add_argument("--limit", type=int, help="Max tasks to evaluate")
    parser.add_argument("--task-ids", nargs="*", help="Specific task IDs to evaluate")
    parser.add_argument("--output", "-o", type=Path, help="Output directory override")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for judge sampling"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode (log all errors)"
    )
    args = parser.parse_args()

    config = load_pipeline_config()

    # Load raw data
    store = BenchmarkStore.load_local("data/raw")
    print(f"Loaded {len(store)} tasks from data/raw/")

    # Select tasks
    if args.task_ids:
        tasks = [store.get(tid) for tid in args.task_ids]
    elif args.source:
        tasks = store.filter(source=args.source)
    else:
        # Load from configured sources
        sources = config["anchor_selection"]["sources"]
        tasks = []
        for s in sources:
            tasks.extend(store.filter(source=s))

    if args.limit:
        tasks = tasks[: args.limit]

    print(f"Evaluating {len(tasks)} tasks...")
    print()

    results = run_anchor_selection(
        tasks,
        config=config,
        output_dir=args.output,
        debug=args.debug,
        seed=args.seed,
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total tasks evaluated: {len(results)}")

    # Top candidates by feasibility
    ranked = sorted(results, key=lambda r: r.feasibility_score, reverse=True)
    print("\nTop 10 by feasibility score:")
    for r in ranked[:10]:
        print(
            f"  {r.task_id:25s}  feas={r.feasibility_score:.1f}  "
            f"best={r.best_ambiguity_type:25s}  risk={r.risk_level}"
        )

    # Distribution by best ambiguity type
    print("\nDistribution by best ambiguity type:")
    from collections import Counter

    type_dist = Counter(r.best_ambiguity_type for r in results)
    for atype, count in type_dist.most_common():
        print(f"  {atype:25s}: {count}")

    # Risk distribution
    risk_dist = Counter(r.risk_level for r in results)
    print("\nRisk distribution:")
    for level, count in risk_dist.most_common():
        print(f"  {level}: {count}")


if __name__ == "__main__":
    main()
