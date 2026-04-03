"""Generic concurrent pipeline runner with JSONL output and progress tracking.

Provides a reusable runner that handles:
  - Concurrent task processing (ThreadPoolExecutor)
  - Incremental JSONL output (thread-safe)
  - Progress display with ETA
  - Error logging to a separate debug file
  - Ordered final output (preserves input order)
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")  # input item type
R = TypeVar("R")  # result type


def run_pipeline(
    items: list[Any],
    process_fn: Callable[[int, Any], dict | None],
    output_path: Path,
    max_workers: int = 8,
    label: str = "Pipeline",
    debug_path: Path | None = None,
) -> list[dict]:
    """Run a concurrent pipeline over a list of items.

    Args:
        items: List of items to process.
        process_fn: Callable(index, item) -> dict result (or None to skip).
            The dict must be JSON-serialisable.
        output_path: Path for the JSONL output file.
        max_workers: Number of concurrent workers.
        label: Display name for progress messages.
        debug_path: Optional path for error/debug log file.

    Returns:
        List of result dicts in original input order (skipped items excluded).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_lock = threading.Lock()
    completed_count = [0]
    results_by_index: dict[int, dict] = {}
    pipeline_start = time.time()

    f_out = open(output_path, "w", encoding="utf-8")
    debug_f = open(debug_path, "w", encoding="utf-8") if debug_path else None

    def _worker(index: int, item: Any) -> None:
        task_start = time.time()
        result = process_fn(index, item)
        task_elapsed = time.time() - task_start

        if result is None:
            with write_lock:
                completed_count[0] += 1
            return

        with write_lock:
            completed_count[0] += 1
            results_by_index[index] = result
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            f_out.flush()

            total_elapsed = time.time() - pipeline_start
            avg = total_elapsed / completed_count[0]
            remaining = len(items) - completed_count[0]
            eta_s = avg * remaining

            # Extract a short summary if the process_fn attached one
            summary = result.pop("__progress__", "")

            # Split summary: first line gets timing, remaining lines print as-is
            lines = summary.split("\n") if summary else [""]
            print(
                f"[{completed_count[0]}/{len(items)}] "
                f"{lines[0]}  "
                f"({task_elapsed:.1f}s)  "
                f"ETA {eta_s / 60:.1f}min"
            )
            for extra_line in lines[1:]:
                if extra_line.strip():
                    print(extra_line)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_worker, i, item): i
            for i, item in enumerate(items)
        }
        for future in futures:
            future.result()

    f_out.close()
    if debug_f:
        debug_f.close()

    total_time = time.time() - pipeline_start

    # Collect results in original order
    results = [results_by_index[i] for i in sorted(results_by_index)]

    # Rewrite file in original order (completion order may differ)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{label} complete in {total_time:.1f}s "
          f"({total_time / len(items):.1f}s/item avg, "
          f"{len(results)}/{len(items)} succeeded)")
    print(f"Output: {output_path}")

    return results
