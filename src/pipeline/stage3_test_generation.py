"""Stage 3 — Reference Solution B + Test Generation.

For each perturbation that passed the entropy gate (Stage 2), generate:
  - ref_solution_b: a complete implementation of interpretation B
  - test_b: a discriminative test suite for interpretation B

The original benchmark provides:
  - canonical_solution (ref_solution_a): implements interpretation A
  - test_code (test_a): original tests for interpretation A

Exclusivity requirement (verified in Stage 4):
  - ref_solution_a passes test_a, fails test_b
  - ref_solution_b passes test_b, fails test_a
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from src.data.model import BenchmarkTask
from src.pipeline.prompts import get_prompt, load_pipeline_config, render_prompt
from src.pipeline.stage2_entropy_gate import Stage2Result
from src.util.llm import LLMClient, ModelConfig
from src.util.parsing import parse_json_response
from src.util.pipeline_runner import run_pipeline


# ── Result types ─────────────────────────────────────────────────────────────

@dataclass
class TestGenOutput:
    """Test generation result for one passed perturbation."""
    task_id: str
    source: str
    entry_point: Optional[str] = None
    library: Optional[str] = None
    best_ambiguity_type: str = ""

    # From the passed entropy result
    generator_model: str = ""
    perturbed_prompt: str = ""
    interpretation_a: str = ""
    interpretation_b: str = ""
    entropy: float = 0.0

    # Generated outputs
    ref_solution_b: str = ""    # implementation of interpretation B
    test_b: str = ""            # test suite for interpretation B
    test_gen_model: str = ""    # model that wrote ref_solution_b + test_b
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TestGenOutput:
        return cls(**d)


# ── Single generation call ──────────────────────────────────────────────────

def generate_ref_and_test(
    client: LLMClient,
    task: BenchmarkTask,
    perturbed_prompt: str,
    interpretation_a: str,
    interpretation_b: str,
    model_alias: str,
    config: dict,
) -> tuple[str, str, Optional[str]]:
    """Generate ref_solution_b and test_b for one perturbation.

    Returns:
        (ref_solution_b, test_b, error_or_none)
    """
    entry_point_instruction = ""
    if task.entry_point:
        entry_point_instruction = (
            f"- Must use `{task.entry_point}` as the function name, "
            f"matching the original signature."
        )

    # For normalized DS1000 tasks, show the original code fragment (not the
    # __SOLUTION__ wrapper) so the LLM understands the actual code pattern,
    # and instruct it to generate fully self-contained ref_solution_b / test_b.
    canonical_solution = task.canonical_solution
    test_code = task.test_code
    if task.source == "ds1000" and task.metadata.get("normalized"):
        canonical_solution = task.metadata.get(
            "original_solution", canonical_solution
        )
        entry_point_instruction += (
            "\n    IMPORTANT — DS1000 FORMAT:"
            "\n    The canonical solution above is a CODE FRAGMENT that relies on"
            " setup variables (imports, DataFrames, arrays, etc.) from a test harness."
            "\n    Your ref_solution_b and test_b MUST be FULLY SELF-CONTAINED:"
            "\n    - ref_solution_b: include ALL imports and define ALL input"
            " variables needed. Do NOT reference undefined variables like df, X,"
            " data, etc."
            "\n    - test_b: include ALL imports, create test data inline, and use"
            " assert statements. Must be runnable as:"
            " exec(ref_solution_b + '\\n' + test_b)."
        )

    system = get_prompt("test_generation.system")
    prompt = render_prompt(
        "test_generation.task",
        perturbed_prompt=perturbed_prompt,
        canonical_solution=canonical_solution,
        test_code=test_code,
        interpretation_a=interpretation_a,
        interpretation_b=interpretation_b,
        entry_point_instruction=entry_point_instruction,
    )

    tg_config = config["test_generation"]
    mc = ModelConfig(
        model=model_alias,
        temperature=tg_config["temperature"],
        max_tokens=tg_config["max_tokens"],
    )

    raw = ""
    try:
        resp = client.call(mc, prompt=prompt, system=system)
        raw = resp.choices[0]
        parsed = parse_json_response(raw)

        ref_solution_b = parsed.get("ref_solution_b", "")
        test_b = parsed.get("test_b", "")

        if not ref_solution_b:
            return "", "", "Empty ref_solution_b in response"
        if not test_b:
            return ref_solution_b, "", "Empty test_b in response"
        return ref_solution_b, test_b, None
    except Exception as e:
        error_msg = str(e)
        if raw:
            error_msg += f"\n    RAW_RESPONSE: {raw[:300]}"
        return "", "", error_msg


# ── Input loading ───────────────────────────────────────────────────────────

def load_stage2_results(path: str | Path) -> list[Stage2Result]:
    """Load Stage 2 results from JSONL."""
    results = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(Stage2Result.from_dict(json.loads(line)))
    return results


def collect_passed_items(
    stage2_results: list[Stage2Result],
) -> list[dict]:
    """Extract all passed entropy results as flat dicts for processing."""
    items = []
    for r in stage2_results:
        for er in r.entropy_results:
            if er.get("passed"):
                items.append({
                    "task_id": r.task_id,
                    "source": r.source,
                    "entry_point": r.entry_point,
                    "library": r.library,
                    "best_ambiguity_type": r.best_ambiguity_type,
                    "generator_model": er["generator_model"],
                    "perturbed_prompt": er["perturbed_prompt"],
                    "interpretation_a": er["interpretation_a"],
                    "interpretation_b": er["interpretation_b"],
                    "entropy": er["entropy"],
                })
    return items


# ── Pipeline runner ─────────────────────────────────────────────────────────

def run_stage3(
    tasks: dict[str, BenchmarkTask],
    stage2_results: list[Stage2Result],
    config: dict | None = None,
    output_dir: str | Path | None = None,
) -> list[TestGenOutput]:
    """Run Stage 3: generate ref_solution_b + test_b for all passed perturbations.

    Args:
        tasks: Mapping of task_id -> BenchmarkTask (raw data lookup).
        stage2_results: Output from Stage 2.
        config: Pipeline config (loaded from pipeline.yaml if None).
        output_dir: Output directory override.

    Returns:
        List of TestGenOutput.
    """
    if config is None:
        config = load_pipeline_config()

    tg_config = config["test_generation"]
    if output_dir is None:
        output_dir = Path(tg_config["output_dir"])
    output_dir = Path(output_dir)

    max_workers = tg_config.get("max_workers", 8)
    use_generator_model = tg_config.get("use_generator_model", True)
    fallback_model = tg_config.get("model", "gpt-5.4")
    client = LLMClient()

    passed_items = collect_passed_items(stage2_results)

    print(f"Stage 3 — Reference Solution B + Test Generation")
    print(f"  Passed perturbations: {len(passed_items)}")
    print(f"  Author: {'same as generator' if use_generator_model else fallback_model}")
    print(f"  Workers: {max_workers}")
    print()

    if not passed_items:
        print("  No passed items to process.")
        return []

    def process_fn(index: int, item: dict) -> dict | None:
        task = tasks.get(item["task_id"])
        if task is None:
            print(f"  SKIP {item['task_id']}: not found in raw data")
            return None

        model = item["generator_model"] if use_generator_model else fallback_model

        ref_solution_b, test_b, error = generate_ref_and_test(
            client, task,
            item["perturbed_prompt"],
            item["interpretation_a"],
            item["interpretation_b"],
            model, config,
        )

        result = TestGenOutput(
            task_id=item["task_id"],
            source=item["source"],
            entry_point=item.get("entry_point"),
            library=item.get("library"),
            best_ambiguity_type=item["best_ambiguity_type"],
            generator_model=item["generator_model"],
            perturbed_prompt=item["perturbed_prompt"],
            interpretation_a=item["interpretation_a"],
            interpretation_b=item["interpretation_b"],
            entropy=item["entropy"],
            ref_solution_b=ref_solution_b,
            test_b=test_b,
            test_gen_model=model,
            error=error,
        ).to_dict()

        status = "OK" if not error else f"ERR: {error[:80]}"
        ref_len = len(ref_solution_b) if ref_solution_b else 0
        test_len = len(test_b) if test_b else 0
        result["__progress__"] = (
            f"{item['task_id']:25s}  "
            f"gen={item['generator_model']:18s}  "
            f"model={model:18s}  "
            f"ref_b={ref_len:4d}  test_b={test_len:4d} chars  "
            f"{status}"
        )
        return result

    raw_results = run_pipeline(
        items=passed_items,
        process_fn=process_fn,
        output_path=output_dir / "stage3_results.jsonl",
        max_workers=max_workers,
        label="Stage 3",
    )

    results = [TestGenOutput.from_dict(r) for r in raw_results]

    # Summary
    ok = sum(1 for r in results if not r.error)
    print(f"\n  Generated: {ok}/{len(results)}")
    if ok:
        avg_ref = sum(len(r.ref_solution_b) for r in results if not r.error) / ok
        avg_test = sum(len(r.test_b) for r in results if not r.error) / ok
        print(f"  Avg ref_solution_b: {avg_ref:.0f} chars")
        print(f"  Avg test_b: {avg_test:.0f} chars")

    return results
