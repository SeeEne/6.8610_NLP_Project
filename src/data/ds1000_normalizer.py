"""DS-1000 format normalizer — converts harness-style tasks to concatenation-friendly format.

DS-1000 tasks use a test harness with [insert] placeholder, exec_context, generate_test_case,
exec_test, and test_execution. This module converts them so that:

    exec(canonical_solution + "\\n" + test_code)

works correctly — matching MBPP/HumanEval execution model.

Conversion strategy:
  - canonical_solution becomes: the original code fragment wrapped in a string variable
    assignment (__SOLUTION__).
  - test_code becomes: the original harness (generate_test_case, exec_test, exec_context,
    test_execution) preserved verbatim, followed by a call to test_execution(__SOLUTION__).

This is minimally invasive — the original DS1000 test logic runs exactly as designed,
we just change how the solution string reaches test_execution().

Matplotlib tasks (image comparison) are skipped — they cannot be meaningfully
converted to a concatenation-based format.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from src.data.model import BenchmarkTask


@dataclass
class NormalizationResult:
    """Result of normalizing one DS1000 task."""

    task: Optional[BenchmarkTask]  # None if skipped
    skipped: bool = False
    skip_reason: str = ""
    error: Optional[str] = None


def _parse_exec_context(test_code: str) -> tuple[str, str, str]:
    """Extract exec_context string and split into (setup, post_insert, full_context).

    Returns:
        (setup_code, post_insert_code, full_exec_context)
    """
    m = re.search(
        r'exec_context\s*=\s*r?("""|\'\'\')(.*?)\1',
        test_code,
        re.DOTALL,
    )
    if not m:
        raise ValueError("Could not find exec_context in test_code")

    full_ctx = m.group(2)
    parts = full_ctx.split("[insert]")
    if len(parts) != 2:
        raise ValueError(
            f"Expected exactly one [insert] in exec_context, found {len(parts) - 1}"
        )

    return parts[0], parts[1], full_ctx


def _build_setup_code(setup_from_ctx: str) -> str:
    """Extract import lines from exec_context setup code."""
    lines = []
    for line in setup_from_ctx.strip().split("\n"):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            lines.append(stripped)
    return "\n".join(lines)


def _wrap_solution_as_string(solution: str) -> str:
    """Wrap a code fragment in a __SOLUTION__ string variable assignment.

    Handles edge cases: triple quotes, backslashes, etc.
    """
    # If safe for r-string (no triple quotes, doesn't end with backslash)
    stripped = solution.rstrip("\n")
    if '"""' not in stripped and not stripped.endswith("\\"):
        return f'__SOLUTION__ = r"""{solution}"""\n'

    # Fallback: use repr() for robust escaping
    return f"__SOLUTION__ = {repr(solution)}\n"


def _has_test_execution(test_code: str) -> bool:
    """Check if test_code defines a test_execution function."""
    return "def test_execution" in test_code


def _has_exec_context(test_code: str) -> bool:
    """Check if test_code defines an exec_context."""
    return "exec_context" in test_code and "[insert]" in test_code


def _extract_skip_filter(test_code: str) -> Optional[str]:
    """Extract any solution filter function (e.g., skip_plt_cmds for Matplotlib)."""
    m = re.search(r"solution = [\"\\n\"].join\(filter\((.*?),", test_code)
    if m:
        return m.group(1).strip()
    return None


def _build_normalized_test(original_test_code: str) -> str:
    """Build normalized test_code by appending test_execution(__SOLUTION__) call.

    The original harness is preserved verbatim. We just add the invocation at the end.
    """
    # Check if there's a solution filter (e.g., Matplotlib's skip_plt_cmds)
    filter_fn = _extract_skip_filter(original_test_code)

    lines = [original_test_code.rstrip()]
    lines.append("")
    lines.append("# --- Normalized execution: run solution through harness ---")
    if filter_fn:
        lines.append(
            f'__SOLUTION_FILTERED__ = "\\n".join('
            f"filter({filter_fn}, __SOLUTION__.split(\"\\n\")))"
        )
        lines.append("test_execution(__SOLUTION_FILTERED__)")
    else:
        lines.append("test_execution(__SOLUTION__)")
    lines.append("")

    return "\n".join(lines)


def normalize_task(task: BenchmarkTask) -> NormalizationResult:
    """Normalize a single DS1000 task to concatenation-friendly format.

    After normalization:
      - canonical_solution: '__SOLUTION__ = r\"\"\"<original code fragment>\"\"\"'
      - test_code: original harness + 'test_execution(__SOLUTION__)'
      - prompt: unchanged

    Execution via sandbox.run(canonical_solution, test_code) will:
      1. Define __SOLUTION__ as a string
      2. Run the original harness
      3. Call test_execution(__SOLUTION__) which inserts it into exec_context

    Args:
        task: A DS1000 BenchmarkTask with harness-style test_code.

    Returns:
        NormalizationResult with the normalized task (or skip/error info).
    """
    if task.source != "ds1000":
        return NormalizationResult(task=task)

    # Skip Matplotlib tasks (image comparison)
    if task.library and task.library.lower() == "matplotlib":
        return NormalizationResult(
            task=None,
            skipped=True,
            skip_reason="Matplotlib tasks use image comparison",
        )

    try:
        if not _has_test_execution(task.test_code):
            return NormalizationResult(
                task=None,
                error="No test_execution function found in test_code",
            )

        if not _has_exec_context(task.test_code):
            return NormalizationResult(
                task=None,
                error="No exec_context with [insert] found in test_code",
            )

        # Parse exec_context for imports (needed by Stage 4)
        setup_code, _, _ = _parse_exec_context(task.test_code)
        setup_imports = _build_setup_code(setup_code)

        # Wrap original solution fragment as a string variable
        normalized_solution = _wrap_solution_as_string(task.canonical_solution)

        # Preserve harness verbatim, append test_execution call
        normalized_test = _build_normalized_test(task.test_code)

        normalized_task = BenchmarkTask(
            task_id=task.task_id,
            source=task.source,
            prompt=task.prompt,
            canonical_solution=normalized_solution,
            test_code=normalized_test,
            entry_point=task.entry_point,
            library=task.library,
            metadata={
                **task.metadata,
                "normalized": True,
                "original_solution": task.canonical_solution,
                "exec_context_imports": setup_imports,
            },
        )

        return NormalizationResult(task=normalized_task)

    except Exception as e:
        return NormalizationResult(
            task=None,
            error=f"{type(e).__name__}: {e}",
        )


def normalize_all(tasks: list[BenchmarkTask]) -> tuple[list[BenchmarkTask], dict]:
    """Normalize a list of DS1000 tasks.

    Returns:
        (normalized_tasks, stats) where stats has counts for success/skip/error.
    """
    stats = {
        "total": 0,
        "success": 0,
        "skipped": 0,
        "errors": 0,
        "error_details": [],
        "skip_reasons": {},
    }
    normalized = []

    for task in tasks:
        if task.source != "ds1000":
            continue
        stats["total"] += 1

        result = normalize_task(task)

        if result.skipped:
            stats["skipped"] += 1
            reason = result.skip_reason
            stats["skip_reasons"][reason] = stats["skip_reasons"].get(reason, 0) + 1
        elif result.error:
            stats["errors"] += 1
            stats["error_details"].append(f"{task.task_id}: {result.error}")
        elif result.task:
            stats["success"] += 1
            normalized.append(result.task)

    return normalized, stats
