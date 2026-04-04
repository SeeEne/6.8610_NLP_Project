"""Per-source loaders that fetch and normalise benchmark tasks.

Each loader returns a list[BenchmarkTask] with a uniform schema.
All downloads are handled by HuggingFace `datasets` (cached locally).
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

from src.data.model import BenchmarkTask


# ── HumanEval ────────────────────────────────────────────────────────────────

def load_humaneval() -> list[BenchmarkTask]:
    """Load all 164 HumanEval tasks.

    Fields used: task_id, prompt, canonical_solution, test, entry_point.
    """
    ds = load_dataset("openai/openai_humaneval", split="test")
    tasks = []
    for row in ds:
        # test field contains a `check(candidate)` function + a call like
        # `check(entry_point)`. We keep it as-is — it's directly executable
        # when concatenated after the solution.
        tasks.append(BenchmarkTask(
            task_id=row["task_id"],
            source="humaneval",
            prompt=row["prompt"],
            canonical_solution=row["canonical_solution"],
            test_code=row["test"],
            entry_point=row["entry_point"],
        ))
    return tasks


# ── MBPP ─────────────────────────────────────────────────────────────────────

def load_mbpp(sanitized: bool = True) -> list[BenchmarkTask]:
    """Load MBPP tasks.

    Args:
        sanitized: If True, load the 427-problem hand-verified "sanitized"
                   subset (recommended). Otherwise load the full 974.
    """
    name = "sanitized" if sanitized else "full"
    ds = load_dataset("google-research-datasets/mbpp", name, split="test")
    tasks = []
    for row in ds:
        # Sanitized uses "prompt", full uses "text"
        prompt_text = row.get("prompt") or row.get("text", "")
        # test_list is a list of assert strings — join into executable block
        test_imports = row.get("test_imports", [])
        test_code = "\n".join(test_imports + row["test_list"])
        tasks.append(BenchmarkTask(
            task_id=f"MBPP/{row['task_id']}",
            source="mbpp",
            prompt=prompt_text,
            canonical_solution=row["code"],
            test_code=test_code,
            metadata={
                "challenge_test_list": row.get("challenge_test_list", []),
            },
        ))
    return tasks


# ── DS-1000 ──────────────────────────────────────────────────────────────────

def load_ds1000() -> list[BenchmarkTask]:
    """Load all 1000 DS-1000 tasks.

    Each task has a prompt, reference_code, and code_context (which contains
    the test harness). Library info is in metadata.
    """
    ds = load_dataset("xlangai/DS-1000", split="test")
    tasks = []
    for row in ds:
        metadata = row.get("metadata", {}) or {}
        # Handle both string metadata and dict metadata
        if isinstance(metadata, str):
            import json
            try:
                metadata = json.loads(metadata)
            except (json.JSONDecodeError, TypeError):
                metadata = {}

        lib = metadata.get("library", "unknown")
        pid = metadata.get("problem_id", row.get("id", ""))
        perturbation = metadata.get("perturbation_type", "")

        tasks.append(BenchmarkTask(
            task_id=f"DS1000/{lib}/{pid}",
            source="ds1000",
            prompt=row["prompt"],
            canonical_solution=row.get("reference_code", ""),
            test_code=row.get("code_context", ""),
            library=lib,
            metadata={
                "perturbation_type": perturbation,
                "perturbation_origin_id": metadata.get("perturbation_origin_id"),
            },
        ))
    return tasks


def load_ds1000_normalized(
    path: str | Path = "data/raw/ds1000_normalized.jsonl",
) -> list[BenchmarkTask]:
    """Load normalized DS-1000 tasks from JSONL.

    These are DS-1000 tasks converted to concatenation-friendly format by
    scripts/normalize_ds1000.py. Execution model matches MBPP/HumanEval:
    exec(canonical_solution + test_code) works directly.

    Matplotlib tasks are excluded (image comparison not convertible).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/normalize_ds1000.py"
        )
    tasks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(BenchmarkTask.from_dict(json.loads(line)))
    return tasks
