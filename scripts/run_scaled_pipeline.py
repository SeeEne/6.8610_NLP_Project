#!/usr/bin/env python3
"""Scaled perturbation pipeline — 10 parallel workers targeting 50 benchmark items.

Each worker handles one (ambiguity_type, risk_level) combination.
Workers pop tasks from a priority queue (sorted by feasibility score),
run Stages 1-4, and stop when they reach their target or exhaust the queue.

Worker allocation (70:30 low:high risk split):
  - 5 low-risk workers  (one per ambiguity type) → 7 items each = 35
  - 5 high-risk workers (one per ambiguity type) → 3 items each = 15
  - Total target: 50 items

Usage:
    python scripts/run_scaled_pipeline.py
    python scripts/run_scaled_pipeline.py --dry-run          # show worker allocations
    python scripts/run_scaled_pipeline.py --low-target 7 --high-target 3
"""

import argparse
import json
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.model import BenchmarkItem, BenchmarkTask
from src.data.store import BenchmarkStore
from src.pipeline.anchor_selection import AnchorResult
from src.pipeline.perturbation import load_anchor_results
from src.pipeline.prompts import load_pipeline_config
from src.pipeline.stage1_perturbation import run_stage1
from src.pipeline.stage2_entropy_gate import run_stage2
from src.pipeline.stage3_test_generation import run_stage3
from src.pipeline.stage4_exclusivity_gate import run_stage4

AMBIGUITY_TYPES = [
    "coreferential",
    "syntactic",
    "scopal",
    "collective_distributive",
    "elliptical",
]


@dataclass
class WorkerResult:
    """Tracks results for one worker."""
    ambiguity_type: str
    risk_level: str
    target: int
    passed: list = field(default_factory=list)     # ExclusivityResult items that passed
    attempted: int = 0
    stage1_total: int = 0
    stage2_passed: int = 0
    stage3_total: int = 0
    stage4_passed: int = 0
    errors: list = field(default_factory=list)


def build_queues(
    anchor_results: list[AnchorResult],
    normalized_ids: set[str],
    min_feasibility: float = 2.0,
) -> dict[tuple[str, str], list[AnchorResult]]:
    """Build priority queues (sorted by feasibility desc) per (type, risk)."""
    queues = defaultdict(list)
    for ar in anchor_results:
        if ar.feasibility_score < min_feasibility:
            continue
        if not ar.best_ambiguity_type:
            continue
        # Only include DS1000 tasks that are normalized, plus all MBPP tasks
        if ar.source == "ds1000" and ar.task_id not in normalized_ids:
            continue
        key = (ar.best_ambiguity_type, ar.risk_level)
        queues[key].append(ar)

    # Sort each queue by feasibility descending
    for key in queues:
        queues[key].sort(key=lambda r: r.feasibility_score, reverse=True)

    return dict(queues)


def run_worker(
    worker_id: int,
    ambiguity_type: str,
    risk_level: str,
    target: int,
    queue: list[AnchorResult],
    task_map: dict[str, BenchmarkTask],
    config: dict,
    output_base: Path,
    print_lock: threading.Lock,
) -> WorkerResult:
    """Run one worker: pop tasks from queue, run Stages 1-4 until target met."""

    tag = f"W{worker_id:02d}|{ambiguity_type[:5]}|{risk_level}"
    result = WorkerResult(
        ambiguity_type=ambiguity_type,
        risk_level=risk_level,
        target=target,
    )

    worker_dir = output_base / f"worker_{worker_id:02d}_{ambiguity_type}_{risk_level}"

    def log(msg: str):
        with print_lock:
            print(f"  [{tag}] {msg}")

    log(f"Starting — target={target}, queue={len(queue)} candidates")

    queue_idx = 0
    while len(result.passed) < target and queue_idx < len(queue):
        anchor = queue[queue_idx]
        queue_idx += 1
        result.attempted += 1

        log(f"Task {anchor.task_id} (feas={anchor.feasibility_score:.1f}) "
            f"[{len(result.passed)}/{target} done, {queue_idx}/{len(queue)} tried]")

        try:
            # Stage 1 — Perturbation generation
            s1_dir = worker_dir / "stage1"
            s1_results = run_stage1(
                tasks=task_map,
                anchor_results=[anchor],
                config=config,
                output_dir=s1_dir,
            )
            result.stage1_total += sum(
                len(r.generations) for r in s1_results
            )

            if not s1_results:
                log(f"  Stage 1: no results")
                continue

            # Stage 2 — Entropy gate
            s2_dir = worker_dir / "stage2"
            s2_results = run_stage2(
                stage1_results=s1_results,
                config=config,
                output_dir=s2_dir,
            )

            any_passed = any(r.any_passed for r in s2_results)
            n_passed = sum(
                1 for r in s2_results
                for er in r.entropy_results if er.get("passed")
            )
            result.stage2_passed += n_passed

            if not any_passed:
                log(f"  Stage 2: 0 passed entropy gate")
                continue

            # Stage 3 — Test generation
            s3_dir = worker_dir / "stage3"
            s3_results = run_stage3(
                tasks=task_map,
                stage2_results=s2_results,
                config=config,
                output_dir=s3_dir,
            )

            valid_s3 = [r for r in s3_results if not r.error]
            result.stage3_total += len(valid_s3)

            if not valid_s3:
                log(f"  Stage 3: no valid results")
                continue

            # Stage 4 — Exclusivity gate
            s4_dir = worker_dir / "stage4"
            s4_results = run_stage4(
                tasks=task_map,
                stage3_results=s3_results,
                config=config,
                output_dir=s4_dir,
            )

            new_passed = [r for r in s4_results if r.passed]
            result.stage4_passed += len(new_passed)

            if new_passed:
                result.passed.extend(new_passed)
                log(f"  PASSED {len(new_passed)} — total {len(result.passed)}/{target}")
            else:
                log(f"  Stage 4: 0 passed exclusivity gate")

        except Exception as e:
            result.errors.append(f"{anchor.task_id}: {e}")
            log(f"  ERROR: {e}")

    log(f"Done — {len(result.passed)}/{target} items collected "
        f"({result.attempted} tasks attempted)")

    return result


def build_benchmark_items(
    worker_results: list[WorkerResult],
    task_map: dict[str, BenchmarkTask],
    existing_count: int = 0,
) -> list[BenchmarkItem]:
    """Convert passed ExclusivityResults into BenchmarkItems."""
    items = []
    item_id = existing_count + 1

    for wr in worker_results:
        for er in wr.passed:
            task = task_map.get(er.task_id)
            if not task:
                continue

            item = BenchmarkItem(
                task_id=f"AMBI/{item_id:03d}",
                anchor_task_id=er.task_id,
                source=task.source,
                prompt=task.prompt,
                canonical_solution=task.canonical_solution,
                test_code=task.test_code,
                entry_point=task.entry_point,
                library=task.library,
                perturbed_prompt=er.perturbed_prompt,
                ambiguity_type=er.best_ambiguity_type,
                risk_level=wr.risk_level,
                interpretation_a=er.interpretation_a,
                interpretation_b=er.interpretation_b,
                ref_solution_a=task.canonical_solution,
                ref_solution_b=er.ref_solution_b,
                test_a=task.test_code,
                test_b=er.test_b,
                quality_gate_a=True,
                quality_gate_b=True,
            )
            items.append(item)
            item_id += 1

    return items


def main():
    parser = argparse.ArgumentParser(description="Scaled perturbation pipeline")
    parser.add_argument("--low-target", type=int, default=7,
                        help="Items per low-risk worker (default: 7)")
    parser.add_argument("--high-target", type=int, default=3,
                        help="Items per high-risk worker (default: 3)")
    parser.add_argument("--min-feasibility", type=float, default=2.0)
    parser.add_argument("--anchor-results", type=Path,
                        default=Path("data/intermediate/anchor_selection/anchor_results.jsonl"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/intermediate/scaled_pipeline"))
    parser.add_argument("--benchmark-out", type=Path,
                        default=Path("data/benchmark/benchmark.jsonl"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Show worker allocations without running")
    parser.add_argument("--max-workers", type=int, default=10,
                        help="Max parallel workers (default: 10)")
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    store = BenchmarkStore.load_local("data/raw")
    task_map = {t.task_id: t for t in store.all_tasks()}
    print(f"  {len(store)} tasks loaded")

    anchor_results = load_anchor_results(args.anchor_results)
    print(f"  {len(anchor_results)} anchor results loaded")

    config = load_pipeline_config()

    # Get normalized DS1000 IDs
    normalized_ids = {
        tid for tid, t in task_map.items()
        if t.source == "ds1000" and t.metadata.get("normalized")
    }

    # Build queues
    queues = build_queues(anchor_results, normalized_ids, args.min_feasibility)

    # Define workers
    workers = []
    for atype in AMBIGUITY_TYPES:
        workers.append((atype, "low", args.low_target))
        workers.append((atype, "high", args.high_target))

    total_target = args.low_target * 5 + args.high_target * 5

    print(f"\n{'='*60}")
    print(f"SCALED PIPELINE — {total_target} items target")
    print(f"{'='*60}")
    print(f"\n{'Worker':<35s} {'Target':>6s} {'Pool':>6s} {'Est':>6s}")
    print("-" * 58)
    for atype, risk, target in workers:
        pool = len(queues.get((atype, risk), []))
        est = int(pool * 0.10)  # ~10% end-to-end yield
        flag = "" if est >= target else "  ⚠ low pool"
        print(f"  {atype:25s} {risk:5s}  {target:>5d}  {pool:>5d}  {est:>5d}{flag}")
    print("-" * 58)
    total_pool = sum(len(q) for q in queues.values())
    print(f"  {'TOTAL':<31s}  {total_target:>5d}  {total_pool:>5d}")

    if args.dry_run:
        print("\n  --dry-run: exiting without running pipeline")
        return

    # Count existing benchmark items
    existing_count = 0
    if args.benchmark_out.exists():
        with open(args.benchmark_out) as f:
            existing_count = sum(1 for line in f if line.strip())
        print(f"\n  Existing benchmark items: {existing_count}")

    # Run workers in parallel
    print(f"\nLaunching {min(len(workers), args.max_workers)} workers...\n")

    print_lock = threading.Lock()
    worker_results: list[WorkerResult] = [None] * len(workers)

    def run_one(idx: int):
        atype, risk, target = workers[idx]
        queue = queues.get((atype, risk), [])
        worker_results[idx] = run_worker(
            worker_id=idx + 1,
            ambiguity_type=atype,
            risk_level=risk,
            target=target,
            queue=queue,
            task_map=task_map,
            config=config,
            output_base=args.output,
            print_lock=print_lock,
        )

    from concurrent.futures import ThreadPoolExecutor
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(run_one, i) for i in range(len(workers))]
        for f in futures:
            f.result()

    elapsed = time.time() - start_time

    # Summary
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE — {elapsed/60:.1f} minutes")
    print(f"{'='*60}")

    print(f"\n{'Worker':<35s} {'Target':>6s} {'Tried':>6s} {'S2':>4s} {'S3':>4s} {'S4':>4s}")
    print("-" * 62)
    total_passed = 0
    for wr in worker_results:
        if wr is None:
            continue
        n = len(wr.passed)
        total_passed += n
        flag = "✓" if n >= wr.target else "✗"
        print(
            f"  {wr.ambiguity_type:25s} {wr.risk_level:5s}  "
            f"{wr.target:>5d}  {wr.attempted:>5d}  "
            f"{wr.stage2_passed:>3d}  {wr.stage3_total:>3d}  "
            f"{wr.stage4_passed:>3d}  {flag}"
        )
    print("-" * 62)
    print(f"  {'TOTAL':<31s}  {total_target:>5d}  "
          f"{'':>5s}  {'':>3s}  {'':>3s}  {total_passed:>3d}")

    # Build and save benchmark items
    if total_passed > 0:
        items = build_benchmark_items(worker_results, task_map, existing_count)
        args.benchmark_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.benchmark_out, "a") as f:
            for item in items:
                f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
        print(f"\n  Appended {len(items)} items to {args.benchmark_out}")
        print(f"  Total benchmark items: {existing_count + len(items)}")

        # Breakdown
        from collections import Counter
        type_counts = Counter(it.ambiguity_type for it in items)
        risk_counts = Counter(it.risk_level for it in items)
        print(f"\n  By ambiguity type:")
        for t in AMBIGUITY_TYPES:
            print(f"    {t:25s}: {type_counts.get(t, 0)}")
        print(f"\n  By risk level:")
        for r in ["low", "high"]:
            print(f"    {r:10s}: {risk_counts.get(r, 0)}")


if __name__ == "__main__":
    main()
