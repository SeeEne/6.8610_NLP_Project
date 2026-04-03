"""Stage 2 — Entropy Gate.

For each Stage 1 perturbation, 5 cheap judge models read ONLY the perturbed
prompt + two interpretations (shuffled as X/Y). Each judge picks which
interpretation the prompt most naturally conveys.

Shannon entropy over the A/B vote distribution measures ambiguity quality:
  - 5-0 split → H=0.00 → no ambiguity → FAIL
  - 4-1 split → H=0.72 → borderline
  - 3-2 split → H=0.97 → real ambiguity → PASS
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from src.pipeline.prompts import get_prompt, load_pipeline_config, render_prompt
from src.pipeline.stage1_perturbation import Stage1Result
from src.util.llm import LLMClient, ModelConfig
from src.util.parsing import parse_json_response
from src.util.pipeline_runner import run_pipeline


# ── Result types ─────────────────────────────────────────────────────────────

@dataclass
class JudgeVote:
    """One judge's vote on which interpretation a prompt conveys."""
    model: str
    choice: str = ""        # "A" or "B" (mapped back from X/Y)
    confidence: str = ""    # "high", "medium", "low"
    raw_choice: str = ""    # "X" or "Y" (before unshuffling)
    error: Optional[str] = None


@dataclass
class EntropyResult:
    """Entropy gate result for one generation within a task."""
    generator_model: str
    perturbed_prompt: str = ""
    interpretation_a: str = ""
    interpretation_b: str = ""

    # Judge votes (mapped to A/B)
    votes: list[dict] = field(default_factory=list)

    # Entropy computation
    count_a: int = 0
    count_b: int = 0
    entropy: float = 0.0
    passed: bool = False

    # Shuffle record: True means A→X, B→Y; False means A→Y, B→X
    a_was_x: bool = True


@dataclass
class Stage2Result:
    """Stage 2 result for one anchor task (entropy results for all generations)."""
    task_id: str
    source: str
    entry_point: Optional[str] = None
    library: Optional[str] = None
    best_ambiguity_type: str = ""

    # Entropy results — one per successful Stage 1 generation
    entropy_results: list[dict] = field(default_factory=list)

    # Summary
    any_passed: bool = False
    best_entropy: float = 0.0
    best_generator: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Stage2Result:
        return cls(**d)


# ── Entropy computation ─────────────────────────────────────────────────────

def compute_entropy(count_a: int, count_b: int) -> float:
    """Binary Shannon entropy from A/B vote counts.

    Returns 0.0 when all votes agree, 1.0 when perfectly split.
    With 5 judges: 5-0 → 0.0, 4-1 → 0.722, 3-2 → 0.971.
    """
    total = count_a + count_b
    if total == 0:
        return 0.0
    p_a = count_a / total
    p_b = count_b / total
    h = 0.0
    if p_a > 0:
        h -= p_a * math.log2(p_a)
    if p_b > 0:
        h -= p_b * math.log2(p_b)
    return round(h, 4)


# ── Single judge call ───────────────────────────────────────────────────────

def judge_single(
    client: LLMClient,
    perturbed_prompt: str,
    interpretation_a: str,
    interpretation_b: str,
    model_alias: str,
    config: dict,
    a_is_x: bool,
) -> JudgeVote:
    """One judge votes on which interpretation the prompt conveys.

    Interpretations are presented as X/Y (shuffled to avoid position bias).
    The vote is mapped back to A/B before returning.
    """
    if a_is_x:
        interp_x, interp_y = interpretation_a, interpretation_b
    else:
        interp_x, interp_y = interpretation_b, interpretation_a

    system = get_prompt("entropy_gate.system")
    prompt = render_prompt(
        "entropy_gate.task",
        perturbed_prompt=perturbed_prompt,
        interpretation_x=interp_x,
        interpretation_y=interp_y,
    )

    mc = ModelConfig(
        model=model_alias,
        temperature=config["entropy_gate"]["temperature"],
        max_tokens=config["entropy_gate"]["max_tokens"],
    )

    raw = ""
    try:
        resp = client.call(mc, prompt=prompt, system=system)
        raw = resp.choices[0]
        parsed = parse_json_response(raw)

        raw_choice = parsed.get("choice", "").upper()
        confidence = parsed.get("confidence", "")

        if raw_choice == "X":
            choice = "A" if a_is_x else "B"
        elif raw_choice == "Y":
            choice = "B" if a_is_x else "A"
        else:
            raise ValueError(f"Invalid choice: {raw_choice}")

        return JudgeVote(
            model=model_alias,
            choice=choice,
            confidence=confidence,
            raw_choice=raw_choice,
        )
    except Exception as e:
        error_msg = str(e)
        if raw:
            error_msg += f"\n    RAW: {raw[:200]}"
        return JudgeVote(model=model_alias, error=error_msg)


# ── Per-generation entropy evaluation ───────────────────────────────────────

def evaluate_generation(
    client: LLMClient,
    perturbed_prompt: str,
    interpretation_a: str,
    interpretation_b: str,
    generator_model: str,
    config: dict,
) -> EntropyResult:
    """Run all judges on a single perturbation concurrently."""
    eg_config = config["entropy_gate"]
    judges = eg_config["judge_models"]
    min_entropy = eg_config["min_entropy"]

    # Deterministic shuffle based on prompt content
    a_is_x = hash(perturbed_prompt) % 2 == 0

    votes: dict[str, JudgeVote] = {}

    def _call(m):
        votes[m] = judge_single(
            client, perturbed_prompt, interpretation_a, interpretation_b,
            m, config, a_is_x,
        )

    with ThreadPoolExecutor(max_workers=len(judges)) as pool:
        for m in judges:
            pool.submit(_call, m)

    vote_list = [votes[m] for m in judges]
    valid = [v for v in vote_list if not v.error]

    count_a = sum(1 for v in valid if v.choice == "A")
    count_b = sum(1 for v in valid if v.choice == "B")
    entropy = compute_entropy(count_a, count_b)

    clean_votes = [
        {k: v for k, v in asdict(vote).items() if k != "raw_choice"}
        for vote in vote_list
    ]

    return EntropyResult(
        generator_model=generator_model,
        perturbed_prompt=perturbed_prompt,
        interpretation_a=interpretation_a,
        interpretation_b=interpretation_b,
        votes=clean_votes,
        count_a=count_a,
        count_b=count_b,
        entropy=entropy,
        passed=entropy >= min_entropy,
        a_was_x=a_is_x,
    )


# ── Pipeline runner ─────────────────────────────────────────────────────────

def load_stage1_results(path: str | Path) -> list[Stage1Result]:
    """Load Stage 1 results from JSONL."""
    results = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(Stage1Result.from_dict(json.loads(line)))
    return results


def run_stage2(
    stage1_results: list[Stage1Result],
    config: dict | None = None,
    output_dir: str | Path | None = None,
) -> list[Stage2Result]:
    """Run Stage 2 entropy gate on all Stage 1 results.

    For each task, evaluates every successful generation from Stage 1.

    Args:
        stage1_results: Output from Stage 1.
        config: Pipeline config (loaded from pipeline.yaml if None).
        output_dir: Output directory override.

    Returns:
        List of Stage2Result in input order.
    """
    if config is None:
        config = load_pipeline_config()

    if output_dir is None:
        output_dir = Path(config["entropy_gate"]["output_dir"])
    output_dir = Path(output_dir)

    eg_config = config["entropy_gate"]
    judges = eg_config["judge_models"]
    min_entropy = eg_config["min_entropy"]
    max_workers = eg_config.get("max_workers", 12)
    client = LLMClient()

    # Count total generations to evaluate
    total_gens = sum(
        1 for r in stage1_results
        for g in r.generations if not g.get("error")
    )

    print(f"Stage 2 — Entropy Gate")
    print(f"  Tasks: {len(stage1_results)}")
    print(f"  Generations to evaluate: {total_gens}")
    print(f"  Judges: {', '.join(judges)}")
    print(f"  Entropy threshold: H >= {min_entropy}")
    print(f"  Workers: {max_workers}")
    print()

    def process_fn(index: int, s1: Stage1Result) -> dict | None:
        ok_gens = [g for g in s1.generations if not g.get("error")]
        if not ok_gens:
            return None

        entropy_results = []
        for g in ok_gens:
            er = evaluate_generation(
                client,
                g["perturbed_prompt"],
                g["interpretation_a"],
                g["interpretation_b"],
                g["model"],
                config,
            )
            entropy_results.append(er)

        clean_entropy = [asdict(er) for er in entropy_results]
        any_passed = any(er.passed for er in entropy_results)
        best_er = max(entropy_results, key=lambda er: er.entropy)

        result = Stage2Result(
            task_id=s1.task_id,
            source=s1.source,
            entry_point=s1.entry_point,
            library=s1.library,
            best_ambiguity_type=s1.best_ambiguity_type,
            entropy_results=clean_entropy,
            any_passed=any_passed,
            best_entropy=best_er.entropy,
            best_generator=best_er.generator_model,
        ).to_dict()

        passed_count = sum(1 for er in entropy_results if er.passed)
        entropy_strs = " | ".join(
            f"{er.generator_model}:H={er.entropy:.2f}{'P' if er.passed else 'F'}"
            for er in entropy_results
        )

        # Build detailed vote breakdown per generation
        vote_details = []
        for er in entropy_results:
            valid_votes = [v for v in er.votes if not v.get("error")]
            vote_str = " ".join(
                f"{v['model']}={v['choice']}" for v in valid_votes
            )
            vote_details.append(
                f"    {er.generator_model:20s}  "
                f"A={er.count_a} B={er.count_b}  H={er.entropy:.3f}  "
                f"{'PASS' if er.passed else 'FAIL'}  "
                f"[{vote_str}]"
            )

        result["__progress__"] = (
            f"{s1.task_id:25s}  "
            f"pass={passed_count}/{len(ok_gens)}  "
            f"[{entropy_strs}]\n" +
            "\n".join(vote_details)
        )
        return result

    raw_results = run_pipeline(
        items=stage1_results,
        process_fn=process_fn,
        output_path=output_dir / "stage2_results.jsonl",
        max_workers=max_workers,
        label="Stage 2",
    )

    results = [Stage2Result.from_dict(r) for r in raw_results]

    # Summary
    total_passed = sum(
        1 for r in results
        for er in r.entropy_results if er.get("passed")
    )
    tasks_with_pass = sum(1 for r in results if r.any_passed)

    print(f"\n  Generations passed: {total_passed}/{total_gens}")
    print(f"  Tasks with >= 1 pass: {tasks_with_pass}/{len(results)}")

    return results
